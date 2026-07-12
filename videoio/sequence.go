package videoio

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	cv "github.com/malcolmston/opencv"
)

// ImageSequenceWriter saves frames as individually numbered image files in a
// directory — the classic "frame0001.png, frame0002.png, …" layout OpenCV
// accepts through a printf-style path. The file format follows the pattern's
// extension: ".png" writes PNG, ".jpg"/".jpeg" writes JPEG. The zero value is
// not usable — obtain one from [NewImageSequenceWriter].
type ImageSequenceWriter struct {
	dir     string
	pattern string
	index   int
	written []string
}

// NewImageSequenceWriter creates a writer that places frames in dir. The pattern
// is a printf template with exactly one integer verb naming each file, for
// example "frame%04d.png" or "img_%d.jpg"; its extension selects the codec.
// index is the number assigned to the first frame (commonly 0 or 1). The
// directory is created if it does not exist.
func NewImageSequenceWriter(dir, pattern string, index int) (*ImageSequenceWriter, error) {
	if err := validatePattern(pattern); err != nil {
		return nil, err
	}
	if dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return nil, fmt.Errorf("videoio: NewImageSequenceWriter mkdir %q: %w", dir, err)
		}
	}
	return &ImageSequenceWriter{dir: dir, pattern: pattern, index: index}, nil
}

// Write encodes frame to the next numbered file and returns its path. Frame
// numbers advance by one on each successful call. It errors on an empty frame
// or if the file cannot be written.
func (w *ImageSequenceWriter) Write(frame *cv.Mat) (string, error) {
	if w == nil {
		return "", fmt.Errorf("videoio: Write on nil ImageSequenceWriter")
	}
	if frame.Empty() {
		return "", fmt.Errorf("videoio: ImageSequenceWriter.Write: empty frame")
	}
	name := fmt.Sprintf(w.pattern, w.index)
	path := filepath.Join(w.dir, name)
	if err := cv.ImWrite(path, frame); err != nil {
		return "", fmt.Errorf("videoio: ImageSequenceWriter.Write %q: %w", path, err)
	}
	w.index++
	w.written = append(w.written, path)
	return path, nil
}

// Count returns the number of frames written so far.
func (w *ImageSequenceWriter) Count() int {
	if w == nil {
		return 0
	}
	return len(w.written)
}

// Files returns the paths written so far, in order.
func (w *ImageSequenceWriter) Files() []string {
	if w == nil {
		return nil
	}
	return append([]string(nil), w.written...)
}

// WriteImageSequence writes every frame to dir using pattern, numbering them
// from start, and returns the paths written. It is the batch counterpart to
// [ImageSequenceWriter]. It errors if frames is empty or the pattern is invalid.
func WriteImageSequence(dir, pattern string, frames []*cv.Mat, start int) ([]string, error) {
	if len(frames) == 0 {
		return nil, fmt.Errorf("videoio: WriteImageSequence: no frames")
	}
	w, err := NewImageSequenceWriter(dir, pattern, start)
	if err != nil {
		return nil, err
	}
	for i, frame := range frames {
		if _, err := w.Write(frame); err != nil {
			return nil, fmt.Errorf("videoio: WriteImageSequence frame %d: %w", i, err)
		}
	}
	return w.Files(), nil
}

// ReadImageSequence loads a numbered image sequence from dir. It reads files
// named by pattern starting at index start and stops at the first index whose
// file is missing, so the sequence must be contiguous. Each file is decoded via
// [cv.ImRead]. It errors if no file exists at the starting index or the pattern
// is invalid.
func ReadImageSequence(dir, pattern string, start int) ([]*cv.Mat, error) {
	if err := validatePattern(pattern); err != nil {
		return nil, err
	}
	var frames []*cv.Mat
	for i := start; ; i++ {
		path := filepath.Join(dir, fmt.Sprintf(pattern, i))
		if _, err := os.Stat(path); err != nil {
			if os.IsNotExist(err) {
				break
			}
			return nil, fmt.Errorf("videoio: ReadImageSequence stat %q: %w", path, err)
		}
		m, err := cv.ImRead(path)
		if err != nil {
			return nil, fmt.Errorf("videoio: ReadImageSequence %q: %w", path, err)
		}
		frames = append(frames, m)
	}
	if len(frames) == 0 {
		return nil, fmt.Errorf("videoio: ReadImageSequence: no file at index %d in %q", start, dir)
	}
	return frames, nil
}

// OpenImageSequence reads a numbered sequence with [ReadImageSequence] and
// returns a [VideoCapture] over the frames, so an on-disk frame directory can be
// streamed like any other source. The delayCentis argument sets the nominal
// per-frame delay reported through CAP_PROP_FPS and used by re-encoders.
func OpenImageSequence(dir, pattern string, start, delayCentis int) (*VideoCapture, error) {
	frames, err := ReadImageSequence(dir, pattern, start)
	if err != nil {
		return nil, err
	}
	if delayCentis < 0 {
		delayCentis = 0
	}
	return newCapture(frames, uniformDelays(len(frames), delayCentis)), nil
}

// validatePattern reports whether pattern is a usable sequence template: it must
// contain exactly one printf integer verb and a recognised image extension.
func validatePattern(pattern string) error {
	if pattern == "" {
		return fmt.Errorf("videoio: empty sequence pattern")
	}
	if n := countIntVerbs(pattern); n != 1 {
		return fmt.Errorf("videoio: sequence pattern %q must contain exactly one %%d verb, found %d", pattern, n)
	}
	switch strings.ToLower(filepath.Ext(pattern)) {
	case ".png", ".jpg", ".jpeg":
		return nil
	default:
		return fmt.Errorf("videoio: sequence pattern %q must end in .png, .jpg or .jpeg", pattern)
	}
}

// countIntVerbs counts the printf integer verbs (%d, with optional width/flags
// such as %04d) in s, treating %% as a literal percent.
func countIntVerbs(s string) int {
	count := 0
	for i := 0; i < len(s); i++ {
		if s[i] != '%' {
			continue
		}
		i++
		if i >= len(s) {
			break
		}
		if s[i] == '%' {
			continue // escaped literal percent
		}
		// Skip flags, width and precision digits/characters up to the verb.
		for i < len(s) && strings.IndexByte("0123456789.+-# ", s[i]) >= 0 {
			i++
		}
		if i < len(s) && s[i] == 'd' {
			count++
		}
	}
	return count
}
