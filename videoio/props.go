package videoio

import (
	cv "github.com/malcolmston/opencv"
)

// PropID identifies a capture or writer property. The values mirror the numeric
// constants of OpenCV's cv::VideoCaptureProperties enumeration so that code
// ported from OpenCV keeps using the same names, and so a property read back
// from one API can be fed to another unchanged.
type PropID int

// Capture and writer property identifiers, matching OpenCV's CAP_PROP_* values.
// Not every backend honours every property; see [VideoCapture.Get],
// [VideoCapture.Set], [VideoWriter.Get] and [VideoWriter.Set] for the subset
// each type supports.
const (
	// CAP_PROP_POS_MSEC is the presentation timestamp of the next frame to be
	// decoded, measured in milliseconds from the start of the clip.
	CAP_PROP_POS_MSEC PropID = 0
	// CAP_PROP_POS_FRAMES is the zero-based index of the next frame to decode.
	// Setting it seeks; see [VideoCapture.SetPosFrames].
	CAP_PROP_POS_FRAMES PropID = 1
	// CAP_PROP_POS_AVI_RATIO is the relative position within the clip in the
	// range [0, 1], where 0 is the first frame and 1 is just past the last.
	CAP_PROP_POS_AVI_RATIO PropID = 2
	// CAP_PROP_FRAME_WIDTH is the frame width in pixels.
	CAP_PROP_FRAME_WIDTH PropID = 3
	// CAP_PROP_FRAME_HEIGHT is the frame height in pixels.
	CAP_PROP_FRAME_HEIGHT PropID = 4
	// CAP_PROP_FPS is the nominal frame rate in frames per second.
	CAP_PROP_FPS PropID = 5
	// CAP_PROP_FOURCC is the four-character codec code packed into a float, as
	// produced by [FourCC].
	CAP_PROP_FOURCC PropID = 6
	// CAP_PROP_FRAME_COUNT is the total number of frames in the clip.
	CAP_PROP_FRAME_COUNT PropID = 7
)

// FourCC packs four ASCII characters into a 32-bit code the way video
// containers identify codecs (for example 'M','J','P','G'). The result matches
// OpenCV's VideoWriter::fourcc and the byte order used in AVI stream headers:
// the first character occupies the least-significant byte.
func FourCC(a, b, c, d byte) uint32 {
	return uint32(a) | uint32(b)<<8 | uint32(c)<<16 | uint32(d)<<24
}

// FourCCString unpacks a code produced by [FourCC] back into its four
// characters.
func FourCCString(code uint32) string {
	return string([]byte{
		byte(code), byte(code >> 8), byte(code >> 16), byte(code >> 24),
	})
}

// fourccGIF is the pseudo-codec code reported for GIF-backed captures.
var fourccGIF = FourCC('G', 'I', 'F', '8')

// FrameGrabber is the read side of the video I/O API, modelled on OpenCV's
// cv::VideoCapture. A grabber yields frames in order through the classic
// grab/retrieve split — [FrameGrabber.Grab] advances to the next frame and
// [FrameGrabber.Retrieve] decodes the most recently grabbed one — or through
// the combined [FrameGrabber.Read]. All frame-backed readers in this package
// ([VideoCapture] and everything that returns one, such as [OpenGIF],
// [OpenAPNG], [OpenAVI] and [OpenImageSequence]) satisfy it.
type FrameGrabber interface {
	// Grab advances to the next frame without decoding it, reporting whether a
	// frame is now available for Retrieve.
	Grab() bool
	// Retrieve returns the frame selected by the most recent successful Grab.
	Retrieve() (*cv.Mat, bool)
	// Read grabs and retrieves the next frame in one step.
	Read() (*cv.Mat, bool)
	// Close releases any resources held by the grabber.
	Close() error
}

var _ FrameGrabber = (*VideoCapture)(nil)

// newCapture builds an in-memory capture from already-decoded frames and their
// per-frame delays (centiseconds). It is the shared constructor used by every
// reader in the package that materialises all frames up front.
func newCapture(frames []*cv.Mat, delays []int) *VideoCapture {
	return &VideoCapture{frames: frames, delays: delays}
}

// Grab advances the capture to the next frame without copying it out, returning
// true while frames remain. It pairs with [VideoCapture.Retrieve]: together
// they perform the same work as [VideoCapture.Read], matching OpenCV's
// grab/retrieve split. Once the frames are exhausted it returns false.
func (c *VideoCapture) Grab() bool {
	if c == nil || c.pos >= len(c.frames) {
		return false
	}
	c.pos++
	return true
}

// Retrieve returns the frame selected by the most recent successful
// [VideoCapture.Grab]. It does not advance the position, so calling it twice in
// a row yields the same frame. Before any Grab, or after the frames are
// exhausted, it returns (nil, false). The returned Mat is the capture's own
// copy; clone it before mutating.
func (c *VideoCapture) Retrieve() (*cv.Mat, bool) {
	if c == nil || c.pos == 0 || c.pos > len(c.frames) {
		return nil, false
	}
	return c.frames[c.pos-1], true
}

// PosFrames returns the zero-based index of the frame that the next
// [VideoCapture.Read] or [VideoCapture.Grab] will return.
func (c *VideoCapture) PosFrames() int {
	if c == nil {
		return 0
	}
	return c.pos
}

// SetPosFrames seeks so that the next read returns frame n. The index is
// clamped to the valid range [0, FrameCount]; seeking to FrameCount leaves the
// capture at end-of-stream. It reports the position actually adopted.
func (c *VideoCapture) SetPosFrames(n int) int {
	if c == nil {
		return 0
	}
	if n < 0 {
		n = 0
	}
	if n > len(c.frames) {
		n = len(c.frames)
	}
	c.pos = n
	return c.pos
}

// posMsec returns the presentation time of the current position in
// milliseconds, summing the delays of all frames already passed. Delays are
// stored in centiseconds, so each contributes ten milliseconds per unit.
func (c *VideoCapture) posMsec() float64 {
	ms := 0.0
	for i := 0; i < c.pos && i < len(c.delays); i++ {
		ms += float64(c.delays[i]) * 10
	}
	return ms
}

// fps derives the nominal frame rate from the first frame's delay. A GIF stores
// delays in centiseconds, so a delay of d yields 100/d frames per second. A
// zero or missing delay reports 0, meaning "unspecified".
func (c *VideoCapture) fps() float64 {
	if len(c.delays) == 0 || c.delays[0] <= 0 {
		return 0
	}
	return 100.0 / float64(c.delays[0])
}

// Get returns the value of property prop, or 0 if the property is unknown or
// the capture is empty. Position properties reflect the current read cursor, so
// they change as frames are consumed.
func (c *VideoCapture) Get(prop PropID) float64 {
	if c == nil {
		return 0
	}
	switch prop {
	case CAP_PROP_POS_FRAMES:
		return float64(c.pos)
	case CAP_PROP_POS_MSEC:
		return c.posMsec()
	case CAP_PROP_POS_AVI_RATIO:
		if len(c.frames) == 0 {
			return 0
		}
		return float64(c.pos) / float64(len(c.frames))
	case CAP_PROP_FRAME_COUNT:
		return float64(len(c.frames))
	case CAP_PROP_FRAME_WIDTH:
		if len(c.frames) == 0 {
			return 0
		}
		return float64(c.frames[0].Cols)
	case CAP_PROP_FRAME_HEIGHT:
		if len(c.frames) == 0 {
			return 0
		}
		return float64(c.frames[0].Rows)
	case CAP_PROP_FPS:
		return c.fps()
	case CAP_PROP_FOURCC:
		return float64(fourccGIF)
	default:
		return 0
	}
}

// Set writes property prop and reports whether the property is settable on a
// capture. Only CAP_PROP_POS_FRAMES (seek, see [VideoCapture.SetPosFrames]),
// CAP_PROP_POS_AVI_RATIO (fractional seek) and CAP_PROP_FPS (rewrite every
// frame delay) are honoured; all other properties are read-only and return
// false.
func (c *VideoCapture) Set(prop PropID, value float64) bool {
	if c == nil {
		return false
	}
	switch prop {
	case CAP_PROP_POS_FRAMES:
		c.SetPosFrames(int(value))
		return true
	case CAP_PROP_POS_AVI_RATIO:
		c.SetPosFrames(int(value * float64(len(c.frames))))
		return true
	case CAP_PROP_FPS:
		if value <= 0 {
			return false
		}
		d := int(100.0/value + 0.5)
		if d < 1 {
			d = 1
		}
		for i := range c.delays {
			c.delays[i] = d
		}
		return true
	default:
		return false
	}
}

// Get returns the value of property prop for the writer, or 0 if it is unknown.
// Supported properties are CAP_PROP_FPS, CAP_PROP_FRAME_COUNT (frames written
// so far), CAP_PROP_FRAME_WIDTH and CAP_PROP_FRAME_HEIGHT (both fixed by the
// first frame, 0 before any frame is written).
func (w *VideoWriter) Get(prop PropID) float64 {
	if w == nil {
		return 0
	}
	switch prop {
	case CAP_PROP_FPS:
		if w.delayCentis <= 0 {
			return 0
		}
		return 100.0 / float64(w.delayCentis)
	case CAP_PROP_FRAME_COUNT:
		return float64(len(w.images))
	case CAP_PROP_FRAME_WIDTH:
		return float64(w.bounds.Dx())
	case CAP_PROP_FRAME_HEIGHT:
		return float64(w.bounds.Dy())
	case CAP_PROP_FOURCC:
		return float64(fourccGIF)
	default:
		return 0
	}
}

// Set writes property prop and reports whether it was applied. Only
// CAP_PROP_FPS is settable, and only before any frame is written; changing the
// rate mid-stream, or setting any other property, returns false.
func (w *VideoWriter) Set(prop PropID, value float64) bool {
	if w == nil || w.released {
		return false
	}
	if prop == CAP_PROP_FPS {
		if value <= 0 || len(w.images) != 0 {
			return false
		}
		d := int(100.0/value + 0.5)
		if d < 1 {
			d = 1
		}
		w.delayCentis = d
		return true
	}
	return false
}
