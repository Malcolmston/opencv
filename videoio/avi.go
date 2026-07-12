package videoio

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"os"

	cv "github.com/malcolmston/opencv"
)

// AVI header flag and stream-index flag bits used by the writer.
const (
	avifHasIndex    = 0x00000010 // AVIF_HASINDEX: file carries an idx1 chunk
	aviifKeyFrame   = 0x00000010 // AVIIF_KEYFRAME: index entry is a key frame
	defaultAVIFPS   = 25.0
	mjpegChunkID    = "00dc" // stream 0, compressed video ("dc")
	mjpegChunkIDAlt = "00db" // stream 0, uncompressed DIB (also accepted on read)
)

// AVIWriter accumulates frames and writes them as a minimal but standards-shaped
// Motion-JPEG AVI file when released. Every frame is JPEG-encoded and stored as
// a chunk inside the movi list of a real RIFF/AVI container, complete with the
// avih main header, a vids/MJPG stream header, a BITMAPINFOHEADER format chunk
// and an idx1 index — so the result is parseable by this package and by
// conventional AVI tooling. The zero value is not usable — obtain one from
// [NewAVIWriter].
type AVIWriter struct {
	path     string
	fps      float64
	width    int
	height   int
	frames   [][]byte
	released bool
}

// NewAVIWriter creates a writer that emits an MJPEG AVI at path when
// [AVIWriter.Release] runs, played back at fps frames per second. A non-positive
// fps is replaced with a sensible default. The frame size is fixed by the first
// frame written. No file is created until Release.
func NewAVIWriter(path string, fps float64) (*AVIWriter, error) {
	if path == "" {
		return nil, fmt.Errorf("videoio: NewAVIWriter: empty path")
	}
	if fps <= 0 {
		fps = defaultAVIFPS
	}
	return &AVIWriter{path: path, fps: fps}, nil
}

// Write JPEG-encodes frame and appends it to the movi stream. The first frame
// fixes the width and height; frames of a different size are still stored, but a
// conforming size keeps the header accurate. It errors on an empty frame or
// after Release.
func (w *AVIWriter) Write(frame *cv.Mat) error {
	if w == nil {
		return fmt.Errorf("videoio: Write on nil AVIWriter")
	}
	if w.released {
		return fmt.Errorf("videoio: AVIWriter.Write after Release")
	}
	if frame.Empty() {
		return fmt.Errorf("videoio: AVIWriter.Write: empty frame")
	}
	jpg, err := cv.IMEncode("jpg", frame)
	if err != nil {
		return fmt.Errorf("videoio: AVIWriter.Write encode: %w", err)
	}
	if len(w.frames) == 0 {
		w.width, w.height = frame.Cols, frame.Rows
	}
	w.frames = append(w.frames, jpg)
	return nil
}

// Release assembles and writes the AVI file, then finalizes the writer. Calling
// it twice, or with no frames written, returns an error.
func (w *AVIWriter) Release() error {
	if w == nil {
		return fmt.Errorf("videoio: Release on nil AVIWriter")
	}
	if w.released {
		return fmt.Errorf("videoio: AVIWriter Release called twice")
	}
	if len(w.frames) == 0 {
		return fmt.Errorf("videoio: AVIWriter Release: no frames written")
	}
	w.released = true

	data := buildAVI(w.frames, w.width, w.height, w.fps)
	if err := os.WriteFile(w.path, data, 0o644); err != nil {
		return fmt.Errorf("videoio: AVIWriter Release write %q: %w", w.path, err)
	}
	return nil
}

// WriteMJPEGAVI encodes frames into a Motion-JPEG AVI at path, played back at
// fps frames per second. It is the batch counterpart to [AVIWriter] and errors
// if frames is empty.
func WriteMJPEGAVI(path string, frames []*cv.Mat, fps float64) error {
	if len(frames) == 0 {
		return fmt.Errorf("videoio: WriteMJPEGAVI: no frames")
	}
	w, err := NewAVIWriter(path, fps)
	if err != nil {
		return err
	}
	for i, frame := range frames {
		if err := w.Write(frame); err != nil {
			return fmt.Errorf("videoio: WriteMJPEGAVI frame %d: %w", i, err)
		}
	}
	return w.Release()
}

// ReadMJPEGAVI parses the MJPEG AVI at path, JPEG-decodes every frame in its
// movi list into a three-channel RGB Mat, and returns the frames together with
// the playback rate recorded in the file header. It walks the RIFF structure
// directly rather than trusting the idx1 index.
func ReadMJPEGAVI(path string) ([]*cv.Mat, float64, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, 0, fmt.Errorf("videoio: ReadMJPEGAVI read %q: %w", path, err)
	}
	jpegs, micros, err := parseAVI(raw)
	if err != nil {
		return nil, 0, fmt.Errorf("videoio: ReadMJPEGAVI %q: %w", path, err)
	}
	frames := make([]*cv.Mat, 0, len(jpegs))
	for i, j := range jpegs {
		m, err := cv.IMDecode(j)
		if err != nil {
			return nil, 0, fmt.Errorf("videoio: ReadMJPEGAVI frame %d: %w", i, err)
		}
		frames = append(frames, m)
	}
	fps := defaultAVIFPS
	if micros > 0 {
		fps = 1e6 / float64(micros)
	}
	return frames, fps, nil
}

// OpenAVI parses the MJPEG AVI at path and returns a [VideoCapture] over its
// frames, with per-frame delays derived from the file's frame rate so that
// CAP_PROP_FPS reads back correctly.
func OpenAVI(path string) (*VideoCapture, error) {
	frames, fps, err := ReadMJPEGAVI(path)
	if err != nil {
		return nil, err
	}
	return newCapture(frames, uniformDelays(len(frames), fpsToCentis(fps))), nil
}

// putU32 and putFourCC write a little-endian uint32 / a literal four-character
// tag to a buffer; AVI, being a RIFF format, is little-endian throughout.
func putU32(buf *bytes.Buffer, v uint32) {
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], v)
	buf.Write(b[:])
}

func putU16(buf *bytes.Buffer, v uint16) {
	var b [2]byte
	binary.LittleEndian.PutUint16(b[:], v)
	buf.Write(b[:])
}

func putFourCC(buf *bytes.Buffer, tag string) {
	buf.WriteString(tag)
}

// buildAVI serialises the JPEG frames into a complete RIFF/AVI byte stream.
func buildAVI(frames [][]byte, width, height int, fps float64) []byte {
	micros := uint32(1e6/fps + 0.5)
	scale := uint32(1000)
	rate := uint32(fps*float64(scale) + 0.5)

	// Build the movi payload first so we can size the surrounding lists, and
	// record each chunk's offset for the idx1 index.
	var movi bytes.Buffer
	putFourCC(&movi, "movi")
	type idxEntry struct {
		offset uint32
		size   uint32
	}
	entries := make([]idxEntry, len(frames))
	for i, f := range frames {
		// Offset is relative to the start of the movi list data (the 'movi'
		// FOURCC), which is the conventional idx1 base.
		entries[i] = idxEntry{offset: uint32(movi.Len()), size: uint32(len(f))}
		putFourCC(&movi, mjpegChunkID)
		putU32(&movi, uint32(len(f)))
		movi.Write(f)
		if len(f)%2 == 1 {
			movi.WriteByte(0) // chunks are word-aligned
		}
	}

	// idx1: one 16-byte entry per frame.
	var idx bytes.Buffer
	for _, e := range entries {
		putFourCC(&idx, mjpegChunkID)
		putU32(&idx, aviifKeyFrame)
		putU32(&idx, e.offset)
		putU32(&idx, e.size)
	}

	// strf: BITMAPINFOHEADER describing MJPG 24-bit frames.
	var strf bytes.Buffer
	putU32(&strf, 40) // biSize
	putU32(&strf, uint32(width))
	putU32(&strf, uint32(height))
	putU16(&strf, 1)  // biPlanes
	putU16(&strf, 24) // biBitCount
	putFourCC(&strf, "MJPG")
	putU32(&strf, uint32(width*height*3)) // biSizeImage
	putU32(&strf, 0)                      // biXPelsPerMeter
	putU32(&strf, 0)                      // biYPelsPerMeter
	putU32(&strf, 0)                      // biClrUsed
	putU32(&strf, 0)                      // biClrImportant

	// strh: video stream header.
	var strh bytes.Buffer
	putFourCC(&strh, "vids")
	putFourCC(&strh, "MJPG")
	putU32(&strh, 0) // dwFlags
	putU16(&strh, 0) // wPriority
	putU16(&strh, 0) // wLanguage
	putU32(&strh, 0) // dwInitialFrames
	putU32(&strh, scale)
	putU32(&strh, rate)
	putU32(&strh, 0) // dwStart
	putU32(&strh, uint32(len(frames)))
	putU32(&strh, uint32(maxLen(frames))) // dwSuggestedBufferSize
	putU32(&strh, 0xFFFFFFFF)             // dwQuality (-1 == default)
	putU32(&strh, 0)                      // dwSampleSize
	putU16(&strh, 0)                      // rcFrame.left
	putU16(&strh, 0)                      // rcFrame.top
	putU16(&strh, uint16(width))          // rcFrame.right
	putU16(&strh, uint16(height))         // rcFrame.bottom

	strl := makeList("strl", strh.Bytes(), chunk("strf", strf.Bytes()))

	// avih: main AVI header.
	var avih bytes.Buffer
	putU32(&avih, micros)
	putU32(&avih, 0) // dwMaxBytesPerSec
	putU32(&avih, 0) // dwPaddingGranularity
	putU32(&avih, avifHasIndex)
	putU32(&avih, uint32(len(frames)))
	putU32(&avih, 0) // dwInitialFrames
	putU32(&avih, 1) // dwStreams
	putU32(&avih, uint32(maxLen(frames)))
	putU32(&avih, uint32(width))
	putU32(&avih, uint32(height))
	putU32(&avih, 0) // dwReserved[0]
	putU32(&avih, 0)
	putU32(&avih, 0)
	putU32(&avih, 0)

	hdrl := buildHdrl(avih.Bytes(), strl)

	var body bytes.Buffer
	putFourCC(&body, "AVI ")
	body.Write(hdrl)
	body.Write(chunk("LIST", movi.Bytes())) // movi already begins with its FOURCC
	body.Write(chunk("idx1", idx.Bytes()))

	var out bytes.Buffer
	putFourCC(&out, "RIFF")
	putU32(&out, uint32(body.Len()))
	out.Write(body.Bytes())
	return out.Bytes()
}

// buildHdrl wraps the avih chunk and the strl list into the hdrl LIST.
func buildHdrl(avih, strl []byte) []byte {
	var inner bytes.Buffer
	inner.Write(chunk("avih", avih))
	inner.Write(strl)
	return chunk("LIST", append([]byte("hdrl"), inner.Bytes()...))
}

// makeList builds a LIST chunk whose body is the list type followed by an
// initial chunk of the given inner type plus any extra pre-serialised chunks.
func makeList(listType string, first []byte, extra ...[]byte) []byte {
	var inner bytes.Buffer
	inner.WriteString(listType)
	inner.Write(chunk("strh", first))
	for _, e := range extra {
		inner.Write(e)
	}
	return chunk("LIST", inner.Bytes())
}

// chunk serialises a single RIFF chunk: four-character id, little-endian size,
// data, and a pad byte when the size is odd.
func chunk(id string, data []byte) []byte {
	var b bytes.Buffer
	b.WriteString(id)
	putU32(&b, uint32(len(data)))
	b.Write(data)
	if len(data)%2 == 1 {
		b.WriteByte(0)
	}
	return b.Bytes()
}

// maxLen returns the length of the largest frame, used to fill in the various
// "suggested buffer size" header fields.
func maxLen(frames [][]byte) int {
	m := 0
	for _, f := range frames {
		if len(f) > m {
			m = len(f)
		}
	}
	return m
}

// parseAVI walks a RIFF/AVI byte stream, returning the JPEG payload of every
// movi frame chunk in order and the microseconds-per-frame from the avih header.
func parseAVI(b []byte) (jpegs [][]byte, microsPerFrame uint32, err error) {
	if len(b) < 12 || string(b[0:4]) != "RIFF" || string(b[8:12]) != "AVI " {
		return nil, 0, fmt.Errorf("not a RIFF/AVI file")
	}
	walkRIFF(b[12:], func(id string, data []byte) {
		switch id {
		case "avih":
			if len(data) >= 4 {
				microsPerFrame = binary.LittleEndian.Uint32(data[0:4])
			}
		case mjpegChunkID, mjpegChunkIDAlt:
			jpegs = append(jpegs, data)
		}
	})
	if len(jpegs) == 0 {
		return nil, 0, fmt.Errorf("no video frames found in movi list")
	}
	return jpegs, microsPerFrame, nil
}

// walkRIFF iterates the chunks in a RIFF body, descending into LIST/RIFF
// containers so that leaf chunks (avih, strh, 00dc, …) are delivered to cb with
// their four-character id and data. Padding bytes are skipped.
func walkRIFF(b []byte, cb func(id string, data []byte)) {
	off := 0
	for off+8 <= len(b) {
		id := string(b[off : off+4])
		size := int(binary.LittleEndian.Uint32(b[off+4 : off+8]))
		off += 8
		if size < 0 || off+size > len(b) {
			return
		}
		data := b[off : off+size]
		if (id == "LIST" || id == "RIFF") && len(data) >= 4 {
			walkRIFF(data[4:], cb) // skip the list type FOURCC
		} else {
			cb(id, data)
		}
		off += size
		if size%2 == 1 {
			off++ // word-alignment padding
		}
	}
}
