package cudacodec

import (
	"fmt"
	"image"
	"path/filepath"
	"strings"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/videoio"
)

// frameSink is the minimal encoder contract shared by every videoio writer this
// package delegates to (GIF, APNG and MJPEG-AVI). Each accepts full RGB [cv.Mat]
// frames and flushes them on release.
type frameSink interface {
	Write(*cv.Mat) error
	Release() error
}

// VideoWriter mirrors cudacodec::VideoWriter, the GPU-accelerated encoder of the
// real module. Here it encodes frames through the sibling
// [github.com/malcolmston/opencv/videoio] package into a standard-library
// container (Motion-JPEG AVI, APNG or GIF), reproducing OpenCV's write/release
// surface while doing all work on the CPU. Obtain one from [CreateVideoWriter]
// or [CreateVideoWriterWithParams].
type VideoWriter struct {
	sink        frameSink
	codec       Codec
	colorFormat ColorFormat
	fps         float64
	size        image.Point
	written     int
	released    bool
}

// CreateVideoWriter opens filename for writing and returns a [VideoWriter],
// mirroring cudacodec::createVideoWriter. The container is chosen from the file
// extension — ".avi" (Motion-JPEG AVI), ".png"/".apng" (APNG) or ".gif"
// (animated GIF) — which is the standard-library substitution this port uses in
// place of NVENC hardware encoding. frameSize is the intended frame size (the
// writer nonetheless adopts the first written frame's actual bounds), codec is
// recorded as intent, fps sets the playback rate and colorFormat is advisory
// metadata. It returns an error for an unsupported extension.
func CreateVideoWriter(filename string, frameSize image.Point, codec Codec, fps float64, colorFormat ColorFormat) (*VideoWriter, error) {
	if fps <= 0 {
		fps = 25
	}
	ext := strings.ToLower(filepath.Ext(filename))
	sink, err := newSink(ext, filename, fps)
	if err != nil {
		return nil, err
	}
	_ = frameSize // requested size is advisory; actual size is fixed by the first written frame
	return &VideoWriter{
		sink:        sink,
		codec:       codec,
		colorFormat: colorFormat,
		fps:         fps,
	}, nil
}

// CreateVideoWriterWithParams is like [CreateVideoWriter] but takes an
// [EncoderParams] bundle, mirroring the params overload of
// cudacodec::createVideoWriter. The frame rate is taken from params.FPS; the
// remaining fields are advisory and recorded for round-tripping.
func CreateVideoWriterWithParams(filename string, frameSize image.Point, codec Codec, colorFormat ColorFormat, params EncoderParams) (*VideoWriter, error) {
	return CreateVideoWriter(filename, frameSize, codec, params.FPS, colorFormat)
}

// newSink builds the videoio writer that backs a given file extension.
func newSink(ext, filename string, fps float64) (frameSink, error) {
	switch ext {
	case ".avi":
		return videoio.NewAVIWriter(filename, fps)
	case ".png", ".apng":
		return videoio.NewAPNGWriter(filename, fpsToCentis(fps))
	case ".gif":
		return videoio.NewGIFWriter(filename, fpsToCentis(fps))
	default:
		return nil, fmt.Errorf("cudacodec: CreateVideoWriter: unsupported extension %q (want .avi, .apng/.png or .gif)", ext)
	}
}

// fpsToCentis converts a frame rate to a per-frame delay in centiseconds for the
// paletted/lossless containers, clamped to at least one centisecond.
func fpsToCentis(fps float64) int {
	if fps <= 0 {
		return 0
	}
	d := int(100.0/fps + 0.5)
	if d < 1 {
		d = 1
	}
	return d
}

// Write encodes one frame, mirroring cudacodec::VideoWriter::write. The frame is
// downloaded from the [GpuMat] and appended to the container. The first frame
// fixes the output size. It returns an error on an empty frame or after
// [VideoWriter.Release].
func (w *VideoWriter) Write(frame *GpuMat) error {
	if w == nil || w.sink == nil {
		return fmt.Errorf("cudacodec: Write on nil VideoWriter")
	}
	if w.released {
		return fmt.Errorf("cudacodec: Write after Release")
	}
	if frame.Empty() {
		return fmt.Errorf("cudacodec: Write: empty frame")
	}
	m := frame.Mat()
	if err := w.sink.Write(m); err != nil {
		return fmt.Errorf("cudacodec: Write: %w", err)
	}
	if w.written == 0 {
		w.size = image.Pt(m.Cols, m.Rows)
	}
	w.written++
	return nil
}

// Release finalizes the container and flushes it to disk, mirroring
// cudacodec::VideoWriter::release. Calling it twice, or releasing a writer with
// no frames, returns an error. After a successful Release the writer must not be
// used again.
func (w *VideoWriter) Release() error {
	if w == nil || w.sink == nil {
		return fmt.Errorf("cudacodec: Release on nil VideoWriter")
	}
	if w.released {
		return fmt.Errorf("cudacodec: Release called twice")
	}
	w.released = true
	if err := w.sink.Release(); err != nil {
		return fmt.Errorf("cudacodec: Release: %w", err)
	}
	return nil
}

// Get returns writer metadata by videoio CAP_PROP_* identifier: CAP_PROP_FPS,
// CAP_PROP_FRAME_COUNT (frames written so far), and CAP_PROP_FRAME_WIDTH /
// CAP_PROP_FRAME_HEIGHT (0 until the first frame is written). Unknown properties
// return 0.
func (w *VideoWriter) Get(prop videoio.PropID) float64 {
	if w == nil {
		return 0
	}
	switch prop {
	case videoio.CAP_PROP_FPS:
		return w.fps
	case videoio.CAP_PROP_FRAME_COUNT:
		return float64(w.written)
	case videoio.CAP_PROP_FRAME_WIDTH:
		return float64(w.size.X)
	case videoio.CAP_PROP_FRAME_HEIGHT:
		return float64(w.size.Y)
	default:
		return 0
	}
}

// Codec returns the codec the writer was created with, echoing the intent
// recorded at construction.
func (w *VideoWriter) Codec() Codec {
	if w == nil {
		return CodecUncompressedRGBA
	}
	return w.codec
}

// ColorFormat returns the advisory color format the writer was created with.
func (w *VideoWriter) ColorFormat() ColorFormat {
	if w == nil {
		return ColorFormatUndefined
	}
	return w.colorFormat
}
