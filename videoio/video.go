package videoio

import (
	"fmt"
	"path/filepath"
	"strings"

	cv "github.com/malcolmston/opencv"
)

// uniformDelays returns a slice of n identical delay values, used when every
// frame of a clip shares one display duration.
func uniformDelays(n, delayCentis int) []int {
	if delayCentis < 0 {
		delayCentis = 0
	}
	d := make([]int, n)
	for i := range d {
		d[i] = delayCentis
	}
	return d
}

// fpsToCentis converts a frame rate to the per-frame delay in centiseconds used
// throughout the package. A non-positive rate yields 0 ("as fast as possible").
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

// WriteVideoFromMats encodes frames to path, choosing the container from the
// file extension: ".gif" writes an animated GIF, ".png"/".apng" writes an APNG,
// and ".avi" writes a Motion-JPEG AVI. fps sets the playback rate (converted to
// per-frame delays for the paletted formats). It errors on an empty frame list
// or an unrecognised extension.
func WriteVideoFromMats(path string, frames []*cv.Mat, fps float64) error {
	if len(frames) == 0 {
		return fmt.Errorf("videoio: WriteVideoFromMats: no frames")
	}
	switch strings.ToLower(filepath.Ext(path)) {
	case ".gif":
		return WriteGIF(path, frames, fpsToCentis(fps))
	case ".png", ".apng":
		return WriteAPNG(path, frames, fpsToCentis(fps))
	case ".avi":
		return WriteMJPEGAVI(path, frames, fps)
	default:
		return fmt.Errorf("videoio: WriteVideoFromMats: unsupported extension %q", filepath.Ext(path))
	}
}

// ReadVideoToMats decodes the clip at path, choosing the container from the file
// extension (".gif", ".png"/".apng" or ".avi"), and returns every frame as a Mat
// together with the clip's frame rate in frames per second. It errors on an
// unrecognised extension or a decode failure.
func ReadVideoToMats(path string) ([]*cv.Mat, float64, error) {
	switch strings.ToLower(filepath.Ext(path)) {
	case ".gif":
		frames, delays, err := ReadGIF(path)
		if err != nil {
			return nil, 0, err
		}
		return frames, centisToFPS(firstDelay(delays)), nil
	case ".png", ".apng":
		frames, delays, err := ReadAPNG(path)
		if err != nil {
			return nil, 0, err
		}
		return frames, centisToFPS(firstDelay(delays)), nil
	case ".avi":
		return ReadMJPEGAVI(path)
	default:
		return nil, 0, fmt.Errorf("videoio: ReadVideoToMats: unsupported extension %q", filepath.Ext(path))
	}
}

// firstDelay returns the first entry of delays, or 0 if empty.
func firstDelay(delays []int) int {
	if len(delays) == 0 {
		return 0
	}
	return delays[0]
}

// centisToFPS converts a per-frame delay in centiseconds to a frame rate. A
// zero delay yields 0 ("unspecified").
func centisToFPS(centis int) float64 {
	if centis <= 0 {
		return 0
	}
	return 100.0 / float64(centis)
}

// ResampleFrames resamples a clip to a constant target frame rate. The input is
// a sequence of frames with per-frame durations in centiseconds; the output is a
// new sequence, sampled at targetFPS by nearest-frame selection, that spans the
// same total duration and whose frames all share the delay 100/targetFPS
// centiseconds. This both up-samples (repeating frames) and down-samples
// (dropping them), which is how a clip is retimed from one playback rate to
// another. The returned Mats are shared with the input (no deep copy). It
// panics if len(frames) != len(delaysCentis) or targetFPS <= 0.
func ResampleFrames(frames []*cv.Mat, delaysCentis []int, targetFPS float64) ([]*cv.Mat, []int) {
	if len(frames) != len(delaysCentis) {
		panic(fmt.Sprintf("videoio: ResampleFrames: %d frames but %d delays", len(frames), len(delaysCentis)))
	}
	if targetFPS <= 0 {
		panic("videoio: ResampleFrames: targetFPS must be positive")
	}
	if len(frames) == 0 {
		return nil, nil
	}

	// Cumulative end time of each source frame, in centiseconds. A zero delay is
	// treated as one centisecond so every frame occupies a non-empty interval.
	ends := make([]float64, len(frames))
	var total float64
	for i, d := range delaysCentis {
		if d <= 0 {
			d = 1
		}
		total += float64(d)
		ends[i] = total
	}

	period := 100.0 / targetFPS // output frame period, centiseconds
	nOut := int(total/period + 0.5)
	if nOut < 1 {
		nOut = 1
	}

	outFrames := make([]*cv.Mat, nOut)
	outDelays := make([]int, nOut)
	outDelay := fpsToCentis(targetFPS)
	src := 0
	for k := 0; k < nOut; k++ {
		// Sample at the centre of the k-th output interval.
		t := (float64(k) + 0.5) * period
		for src < len(ends)-1 && t > ends[src] {
			src++
		}
		outFrames[k] = frames[src]
		outDelays[k] = outDelay
	}
	return outFrames, outDelays
}

// ResampleCapture resamples a capture's frames to targetFPS and returns a fresh
// [VideoCapture] positioned at the start, so a clip opened at one rate can be
// replayed at another. It reads the source frames and delays as they currently
// stand; it does not consume or modify the input capture's read position beyond
// what it needs.
func ResampleCapture(c *VideoCapture, targetFPS float64) *VideoCapture {
	if c == nil {
		return newCapture(nil, nil)
	}
	frames, delays := ResampleFrames(c.frames, c.delays, targetFPS)
	return newCapture(frames, delays)
}
