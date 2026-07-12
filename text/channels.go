package text

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// NMChannelMode selects the channel decomposition produced by
// [ComputeNMChannels], mirroring OpenCV's ERFILTER_NM_* modes.
type NMChannelMode int

const (
	// NMRGBLGrad decomposes a colour image into its red, green and blue
	// channels, its lightness (grayscale) and the gradient magnitude of the
	// lightness. This is OpenCV's default for the Neumann–Matas detector.
	NMRGBLGrad NMChannelMode = iota
	// NMIHSGrad decomposes into intensity (grayscale), hue, saturation and the
	// gradient magnitude of the intensity.
	NMIHSGrad
)

// ComputeNMChannels reproduces the channel decomposition that OpenCV's
// Neumann–Matas scene-text detector runs its extremal-region search over. It
// returns a slice of single-channel [cv.Mat]s, one per channel of the selected
// mode. The extremal-region search is then run independently on each channel and
// the results merged.
//
// For [NMRGBLGrad] the channels are {R, G, B, Lightness, Gradient}; for
// [NMIHSGrad] they are {Intensity, Hue, Saturation, Gradient}. img must be
// three-channel RGB; it panics otherwise.
func ComputeNMChannels(img *cv.Mat, mode NMChannelMode) []*cv.Mat {
	if img.Channels != 3 {
		panic("text: ComputeNMChannels requires a 3-channel RGB image")
	}
	gray := cv.CvtColor(img, cv.ColorRGB2Gray)
	grad := gradientMagnitude(gray)

	switch mode {
	case NMRGBLGrad:
		planes := img.Split() // R, G, B
		return []*cv.Mat{planes[0], planes[1], planes[2], gray, grad}
	case NMIHSGrad:
		hsv := cv.CvtColor(img, cv.ColorRGB2HSV)
		hsvPlanes := hsv.Split() // H, S, V
		return []*cv.Mat{gray, hsvPlanes[0], hsvPlanes[1], grad}
	default:
		panic("text: ComputeNMChannels unknown mode")
	}
}

// gradientMagnitude returns the Sobel gradient magnitude of a single-channel
// image, clamped to the 8-bit range. Borders use edge replication.
func gradientMagnitude(gray *cv.Mat) *cv.Mat {
	rows, cols := gray.Rows, gray.Cols
	out := cv.NewMat(rows, cols, 1)
	at := func(y, x int) float64 {
		if y < 0 {
			y = 0
		} else if y >= rows {
			y = rows - 1
		}
		if x < 0 {
			x = 0
		} else if x >= cols {
			x = cols - 1
		}
		return float64(gray.Data[y*cols+x])
	}
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			gx := -at(y-1, x-1) - 2*at(y, x-1) - at(y+1, x-1) +
				at(y-1, x+1) + 2*at(y, x+1) + at(y+1, x+1)
			gy := -at(y-1, x-1) - 2*at(y-1, x) - at(y-1, x+1) +
				at(y+1, x-1) + 2*at(y+1, x) + at(y+1, x+1)
			mag := math.Hypot(gx, gy)
			if mag > 255 {
				mag = 255
			}
			out.Data[y*cols+x] = uint8(mag + 0.5)
		}
	}
	return out
}
