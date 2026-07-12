package videoio

import (
	"fmt"
	"image"
	"image/color"
	"image/color/palette"
	"image/draw"
	"image/gif"
	"os"

	cv "github.com/malcolmston/opencv"
)

// quantPalette is the fixed palette every encoded frame is mapped onto. The
// 216-colour web-safe palette has channel levels evenly spaced 51 apart, which
// bounds the nearest-colour error to about 26 per channel and — combined with
// non-dithered mapping — makes encoding fully deterministic.
var quantPalette = color.Palette(palette.WebSafe)

// VideoCapture reads the frames of an animated GIF as a sequence of Mats. All
// frames are decoded eagerly when the capture is opened, then handed out one at
// a time by [VideoCapture.Read]. A VideoCapture holds no operating-system
// resources once opened; [VideoCapture.Close] simply releases the decoded
// frames. The zero value is not usable — obtain one from [OpenGIF].
type VideoCapture struct {
	frames []*cv.Mat
	delays []int
	pos    int
}

// OpenGIF opens the animated (or single-frame) GIF at path and decodes every
// frame into memory. It returns an error if the file cannot be read or does not
// contain a valid GIF.
func OpenGIF(path string) (*VideoCapture, error) {
	frames, delays, err := ReadGIF(path)
	if err != nil {
		return nil, err
	}
	return &VideoCapture{frames: frames, delays: delays}, nil
}

// Read returns the next frame and true, advancing the internal position. Once
// every frame has been returned it yields (nil, false) on every subsequent
// call. The returned Mat is the capture's own copy; callers that intend to
// mutate it should [cv.Mat.Clone] it first.
func (c *VideoCapture) Read() (*cv.Mat, bool) {
	if c == nil || c.pos >= len(c.frames) {
		return nil, false
	}
	frame := c.frames[c.pos]
	c.pos++
	return frame, true
}

// Frames returns all decoded frames in order. The returned slice is the
// capture's own backing slice and shares its Mats; do not mutate it in place.
// Reading it does not affect the position used by [VideoCapture.Read].
func (c *VideoCapture) Frames() []*cv.Mat {
	return c.frames
}

// Delays returns the per-frame display durations in centiseconds (hundredths of
// a second), one entry per frame and in the same order as [VideoCapture.Frames].
func (c *VideoCapture) Delays() []int {
	return c.delays
}

// FrameCount returns the total number of decoded frames.
func (c *VideoCapture) FrameCount() int {
	return len(c.frames)
}

// Close releases the decoded frames and resets the capture. After Close the
// capture reports zero frames and [VideoCapture.Read] returns (nil, false). It
// never fails and always returns nil; the error result exists to match the
// idiomatic io.Closer shape.
func (c *VideoCapture) Close() error {
	if c == nil {
		return nil
	}
	c.frames = nil
	c.delays = nil
	c.pos = 0
	return nil
}

// VideoWriter accumulates frames and encodes them as a single animated GIF when
// released. Frames are quantized to the web-safe palette as they arrive. The
// canvas size is fixed by the first frame written: later frames are placed at
// the origin and clipped to those bounds. The zero value is not usable — obtain
// one from [NewGIFWriter].
type VideoWriter struct {
	path        string
	delayCentis int
	bounds      image.Rectangle
	images      []*image.Paletted
	delays      []int
	released    bool
}

// NewGIFWriter creates a writer that will encode all frames given to
// [VideoWriter.Write] into an animated GIF at path when [VideoWriter.Release] is
// called. delayCentis is the display duration of each frame in centiseconds
// (hundredths of a second); for example 10 yields roughly ten frames per
// second. A non-positive delayCentis is clamped to 0, which most viewers treat
// as "as fast as possible". No file is created until Release runs.
func NewGIFWriter(path string, delayCentis int) (*VideoWriter, error) {
	if path == "" {
		return nil, fmt.Errorf("videoio: NewGIFWriter: empty path")
	}
	if delayCentis < 0 {
		delayCentis = 0
	}
	return &VideoWriter{path: path, delayCentis: delayCentis}, nil
}

// Write quantizes frame and appends it to the animation. The first frame fixes
// the output size; every later frame is drawn at the canvas origin and any part
// extending past the first frame's bounds is clipped. It returns an error if
// the frame is empty or the writer has already been released.
func (w *VideoWriter) Write(frame *cv.Mat) error {
	if w == nil {
		return fmt.Errorf("videoio: Write on nil writer")
	}
	if w.released {
		return fmt.Errorf("videoio: Write after Release")
	}
	if frame.Empty() {
		return fmt.Errorf("videoio: Write: empty frame")
	}
	src := frame.ToImage()
	if len(w.images) == 0 {
		w.bounds = image.Rect(0, 0, frame.Cols, frame.Rows)
	}
	paletted := image.NewPaletted(w.bounds, quantPalette)
	// draw.Src maps each pixel to its nearest palette colour with no dithering,
	// which keeps the output deterministic. Placing src at the canvas origin and
	// intersecting bounds handles frames larger or smaller than the first one.
	draw.Draw(paletted, w.bounds.Intersect(src.Bounds()), src, src.Bounds().Min, draw.Src)
	w.images = append(w.images, paletted)
	w.delays = append(w.delays, w.delayCentis)
	return nil
}

// Release encodes every written frame to the destination GIF and finalizes the
// writer. It uses disposal method "none" for all frames, so each frame is drawn
// over the previous one, and sets an infinite loop count. Release is idempotent
// only in that calling it a second time returns an error; a writer with no
// frames also returns an error. After a successful Release the writer must not
// be used again.
func (w *VideoWriter) Release() error {
	if w == nil {
		return fmt.Errorf("videoio: Release on nil writer")
	}
	if w.released {
		return fmt.Errorf("videoio: Release called twice")
	}
	if len(w.images) == 0 {
		return fmt.Errorf("videoio: Release: no frames written")
	}
	w.released = true

	disposal := make([]byte, len(w.images))
	for i := range disposal {
		disposal[i] = gif.DisposalNone
	}
	g := &gif.GIF{
		Image:     w.images,
		Delay:     w.delays,
		Disposal:  disposal,
		LoopCount: 0, // loop forever
		Config: image.Config{
			ColorModel: quantPalette,
			Width:      w.bounds.Dx(),
			Height:     w.bounds.Dy(),
		},
	}

	f, err := os.Create(w.path)
	if err != nil {
		return fmt.Errorf("videoio: Release create %q: %w", w.path, err)
	}
	if err := gif.EncodeAll(f, g); err != nil {
		f.Close()
		return fmt.Errorf("videoio: Release encode %q: %w", w.path, err)
	}
	if err := f.Close(); err != nil {
		return fmt.Errorf("videoio: Release close %q: %w", w.path, err)
	}
	return nil
}

// ReadGIF decodes the GIF at path and returns every frame as a Mat together
// with the matching per-frame delays in centiseconds. Partial frames and GIF
// disposal methods are honoured: each stored sub-image is composited onto a
// full-size canvas so the returned Mats are complete, canvas-sized frames. The
// frames are three-channel RGB.
func ReadGIF(path string) ([]*cv.Mat, []int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, fmt.Errorf("videoio: ReadGIF open %q: %w", path, err)
	}
	defer f.Close()

	g, err := gif.DecodeAll(f)
	if err != nil {
		return nil, nil, fmt.Errorf("videoio: ReadGIF decode %q: %w", path, err)
	}
	if len(g.Image) == 0 {
		return nil, nil, fmt.Errorf("videoio: ReadGIF %q: no frames", path)
	}

	bounds := image.Rect(0, 0, g.Config.Width, g.Config.Height)
	if bounds.Empty() {
		bounds = g.Image[0].Bounds()
	}

	canvas := image.NewRGBA(bounds)
	frames := make([]*cv.Mat, 0, len(g.Image))
	delays := make([]int, 0, len(g.Image))
	var saved *image.RGBA

	for i, src := range g.Image {
		disposal := byte(0)
		if i < len(g.Disposal) {
			disposal = g.Disposal[i]
		}
		if disposal == gif.DisposalPrevious {
			saved = cloneRGBA(canvas)
		}

		// draw.Over respects the frame's transparent palette index, letting
		// earlier content show through where this frame is transparent.
		draw.Draw(canvas, src.Bounds(), src, src.Bounds().Min, draw.Over)

		// cv.FromImage copies the samples out immediately, so it is safe to keep
		// reusing and mutating canvas for subsequent frames.
		frames = append(frames, cv.FromImage(canvas))

		delay := 0
		if i < len(g.Delay) {
			delay = g.Delay[i]
		}
		delays = append(delays, delay)

		switch disposal {
		case gif.DisposalBackground:
			// Restore the frame's rectangle to the (transparent) background.
			draw.Draw(canvas, src.Bounds(), image.Transparent, image.Point{}, draw.Src)
		case gif.DisposalPrevious:
			if saved != nil {
				draw.Draw(canvas, canvas.Bounds(), saved, canvas.Bounds().Min, draw.Src)
			}
		}
	}
	return frames, delays, nil
}

// WriteGIF encodes frames into an animated GIF at path, giving every frame the
// same delay of delayCentis centiseconds. It is a convenience wrapper around
// [NewGIFWriter], [VideoWriter.Write] and [VideoWriter.Release]. The output size
// is taken from the first frame; later frames are placed at the origin and
// clipped. It returns an error if frames is empty.
func WriteGIF(path string, frames []*cv.Mat, delayCentis int) error {
	if len(frames) == 0 {
		return fmt.Errorf("videoio: WriteGIF: no frames")
	}
	w, err := NewGIFWriter(path, delayCentis)
	if err != nil {
		return err
	}
	for i, frame := range frames {
		if err := w.Write(frame); err != nil {
			return fmt.Errorf("videoio: WriteGIF frame %d: %w", i, err)
		}
	}
	return w.Release()
}

// cloneRGBA returns an independent deep copy of src, used to snapshot the canvas
// for the GIF "restore to previous" disposal method.
func cloneRGBA(src *image.RGBA) *image.RGBA {
	dst := image.NewRGBA(src.Bounds())
	copy(dst.Pix, src.Pix)
	return dst
}
