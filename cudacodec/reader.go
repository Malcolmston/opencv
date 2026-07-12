package cudacodec

import (
	"fmt"
	"image"
	"path/filepath"
	"strings"

	"github.com/malcolmston/opencv/videoio"
)

// VideoReader mirrors cudacodec::VideoReader, the GPU-accelerated decoder of the
// real module. Here it decodes a standard-library container
// (Motion-JPEG AVI, APNG or GIF) through the sibling
// [github.com/malcolmston/opencv/videoio] package and hands frames out as
// [GpuMat] values, reproducing OpenCV's grab/retrieve/nextFrame surface. Obtain
// one from [CreateVideoReader] and free it with [VideoReader.Release].
type VideoReader struct {
	cap    *videoio.VideoCapture
	format FormatInfo
}

// CreateVideoReader opens the clip at filename and returns a [VideoReader],
// mirroring cudacodec::createVideoReader. The container is selected from the file
// extension: ".avi" (Motion-JPEG AVI), ".png"/".apng" (APNG) and ".gif"
// (animated GIF) are supported — the standard-library substitutions this port
// uses in place of NVDEC hardware decoding. It returns an error for an
// unsupported extension or a decode failure.
func CreateVideoReader(filename string) (*VideoReader, error) {
	ext := strings.ToLower(filepath.Ext(filename))
	var (
		cap   *videoio.VideoCapture
		err   error
		codec = CodecUncompressedRGBA
	)
	switch ext {
	case ".avi":
		cap, err = videoio.OpenAVI(filename)
		codec = CodecJPEG
	case ".png", ".apng":
		cap, err = videoio.OpenAPNG(filename)
	case ".gif":
		cap, err = videoio.OpenGIF(filename)
	default:
		return nil, fmt.Errorf("cudacodec: CreateVideoReader: unsupported extension %q (want .avi, .apng/.png or .gif)", ext)
	}
	if err != nil {
		return nil, fmt.Errorf("cudacodec: CreateVideoReader %q: %w", filename, err)
	}

	r := &VideoReader{cap: cap}
	r.format = buildFormatInfo(cap, codec)
	return r, nil
}

// buildFormatInfo derives a [FormatInfo] from an opened capture.
func buildFormatInfo(cap *videoio.VideoCapture, codec Codec) FormatInfo {
	w := int(cap.Get(videoio.CAP_PROP_FRAME_WIDTH))
	h := int(cap.Get(videoio.CAP_PROP_FRAME_HEIGHT))
	chroma := ChromaYUV420
	if codec == CodecUncompressedRGBA {
		chroma = ChromaYUV444
	}
	return FormatInfo{
		Codec:       codec,
		Chroma:      chroma,
		Width:       w,
		Height:      h,
		DisplayArea: image.Rect(0, 0, w, h),
		FPS:         cap.Get(videoio.CAP_PROP_FPS),
		NumFrames:   int(cap.Get(videoio.CAP_PROP_FRAME_COUNT)),
		Valid:       cap.FrameCount() > 0,
	}
}

// Format returns the [FormatInfo] describing the open stream, mirroring
// cudacodec::VideoReader::format. The values are fixed when the reader is
// created and do not change as frames are consumed.
func (r *VideoReader) Format() FormatInfo {
	if r == nil {
		return FormatInfo{}
	}
	return r.format
}

// NextFrame decodes the next frame into dst and reports whether a frame was
// produced, mirroring cudacodec::VideoReader::nextFrame. When it returns false
// the stream is exhausted and dst is left unchanged. The variadic stream
// argument is accepted for API compatibility and ignored (see [Stream]). It
// panics if dst is nil.
func (r *VideoReader) NextFrame(dst *GpuMat, _ ...*Stream) bool {
	if dst == nil {
		panic("cudacodec: NextFrame requires a non-nil destination GpuMat")
	}
	if r == nil || r.cap == nil {
		return false
	}
	frame, ok := r.cap.Read()
	if !ok {
		return false
	}
	dst.Upload(frame)
	return true
}

// Grab advances to the next frame without decoding it into a destination,
// mirroring cudacodec::VideoReader::grab. It reports whether a frame is now
// available for [VideoReader.Retrieve]. The variadic stream argument is ignored.
func (r *VideoReader) Grab(_ ...*Stream) bool {
	if r == nil || r.cap == nil {
		return false
	}
	return r.cap.Grab()
}

// Retrieve returns the frame selected by the most recent successful
// [VideoReader.Grab], mirroring cudacodec::VideoReader::retrieve. It fills dst
// and reports success; before any Grab, or once the stream is exhausted, it
// returns false and leaves dst unchanged. It panics if dst is nil.
func (r *VideoReader) Retrieve(dst *GpuMat, _ ...*Stream) bool {
	if dst == nil {
		panic("cudacodec: Retrieve requires a non-nil destination GpuMat")
	}
	if r == nil || r.cap == nil {
		return false
	}
	frame, ok := r.cap.Retrieve()
	if !ok {
		return false
	}
	dst.Upload(frame)
	return true
}

// Get returns the value of capture property prop, mirroring
// cudacodec::VideoReader::get. The property identifiers are videoio's CAP_PROP_*
// constants; position properties reflect how many frames have been consumed.
func (r *VideoReader) Get(prop videoio.PropID) float64 {
	if r == nil || r.cap == nil {
		return 0
	}
	return r.cap.Get(prop)
}

// FrameCount returns the total number of frames in the stream.
func (r *VideoReader) FrameCount() int {
	if r == nil || r.cap == nil {
		return 0
	}
	return r.cap.FrameCount()
}

// Release frees the reader and its decoded frames, mirroring the destruction of
// a cudacodec::VideoReader. It is safe to call more than once and always returns
// nil; the error result matches OpenCV's Release shape.
func (r *VideoReader) Release() error {
	if r == nil || r.cap == nil {
		return nil
	}
	err := r.cap.Close()
	r.cap = nil
	return err
}
