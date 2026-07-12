package dnn_superres

import (
	"fmt"
	"math"

	cv "github.com/malcolmston/opencv"
)

// validate checks a source image and scale factor shared by every upsampler.
func validate(src *cv.Mat, scale int) error {
	if src == nil || src.Empty() {
		return fmt.Errorf("dnn_superres: source image is empty")
	}
	if scale != 2 && scale != 3 && scale != 4 {
		return fmt.Errorf("dnn_superres: unsupported scale %d (want 2, 3 or 4)", scale)
	}
	return nil
}

// UpsampleNearest enlarges src by an integer scale (2, 3 or 4) using
// nearest-neighbour sampling. It is the fastest and blockiest method and
// reproduces every source sample exactly, so hard edges stay perfectly sharp at
// the cost of visible pixelation. It returns an error for an empty image or an
// unsupported scale.
func UpsampleNearest(src *cv.Mat, scale int) (*cv.Mat, error) {
	if err := validate(src, scale); err != nil {
		return nil, err
	}
	dstW, dstH := src.Cols*scale, src.Rows*scale
	ch := src.Channels
	dst := cv.NewMat(dstH, dstW, ch)
	for y := 0; y < dstH; y++ {
		sy := clampInt(y/scale, 0, src.Rows-1)
		for x := 0; x < dstW; x++ {
			sx := clampInt(x/scale, 0, src.Cols-1)
			si := (sy*src.Cols + sx) * ch
			di := (y*dstW + x) * ch
			copy(dst.Data[di:di+ch], src.Data[si:si+ch])
		}
	}
	return dst, nil
}

// triangle is the 1-D linear (bilinear) interpolation kernel, support radius 1.
func triangle(t float64) float64 {
	t = math.Abs(t)
	if t < 1 {
		return 1 - t
	}
	return 0
}

// UpsampleBilinear enlarges src by an integer scale (2, 3 or 4) using bilinear
// interpolation — a separable 2-tap linear resample. It produces smooth,
// artefact-free gradients but softens fine detail. It returns an error for an
// empty image or an unsupported scale.
func UpsampleBilinear(src *cv.Mat, scale int) (*cv.Mat, error) {
	if err := validate(src, scale); err != nil {
		return nil, err
	}
	return resampleSeparable(src, src.Cols*scale, src.Rows*scale, triangle, 1), nil
}

// UpsampleBicubic enlarges src by an integer scale (2, 3 or 4) using Keys /
// Catmull-Rom bicubic convolution (a = -0.5), the 4-tap separable cubic used by
// cv2.INTER_CUBIC. It preserves edges better than bilinear with mild, controlled
// overshoot. It returns an error for an empty image or an unsupported scale.
func UpsampleBicubic(src *cv.Mat, scale int) (*cv.Mat, error) {
	if err := validate(src, scale); err != nil {
		return nil, err
	}
	return resampleSeparable(src, src.Cols*scale, src.Rows*scale, keysCubic, 2), nil
}

// UpsampleLanczos enlarges src by an integer scale (2, 3 or 4) using Lanczos-4
// windowed-sinc interpolation (8-tap separable), matching cv2.INTER_LANCZOS4. It
// gives the crispest classical result and the strongest high-frequency
// retention, at the cost of visible ringing near hard edges. It returns an error
// for an empty image or an unsupported scale.
func UpsampleLanczos(src *cv.Mat, scale int) (*cv.Mat, error) {
	if err := validate(src, scale); err != nil {
		return nil, err
	}
	return resampleSeparable(src, src.Cols*scale, src.Rows*scale, lanczos4, 4), nil
}

// UpsampleEdgeDirected enlarges src by an integer scale (2, 3 or 4) with an
// edge-directed method (a NEDI-lite / edge-guided cubic). A bicubic base image
// is computed first, then, guided by the source luminance gradient, each strong
// edge pixel is smoothed only along the edge tangent (perpendicular to the
// gradient). This suppresses the staircase/jaggy artefacts that plain
// interpolation leaves on diagonal edges without blurring across the edge, so
// contours look markedly cleaner than bicubic while flat regions are left
// untouched. It is fully classical and uses no learned weights. It returns an
// error for an empty image or an unsupported scale.
func UpsampleEdgeDirected(src *cv.Mat, scale int) (*cv.Mat, error) {
	if err := validate(src, scale); err != nil {
		return nil, err
	}
	base, err := UpsampleBicubic(src, scale)
	if err != nil {
		return nil, err
	}
	return edgeGuidedSmooth(src, base, scale), nil
}

// grayscale returns a single-channel luminance view of src used only for edge
// analysis. For 1-channel input it is the data itself; for 3-channel input it
// uses the Rec.601 luma weights; otherwise it averages all channels.
func grayscale(src *cv.Mat) *cv.Mat {
	if src.Channels == 1 {
		return src
	}
	g := cv.NewMat(src.Rows, src.Cols, 1)
	for i := 0; i < src.Total(); i++ {
		base := i * src.Channels
		var v float64
		if src.Channels == 3 {
			v = 0.299*float64(src.Data[base]) + 0.587*float64(src.Data[base+1]) + 0.114*float64(src.Data[base+2])
		} else {
			var sum int
			for c := 0; c < src.Channels; c++ {
				sum += int(src.Data[base+c])
			}
			v = float64(sum) / float64(src.Channels)
		}
		g.Data[i] = clampByte(v)
	}
	return g
}

// edgeGuidedSmooth refines a bicubic base image using source edge orientation.
// Gradients are estimated on the source luminance with cv.SobelFloat; at each
// destination pixel whose corresponding source location has gradient magnitude
// above an adaptive threshold, the base image is resampled at three points along
// the local edge tangent and blended, with the blend weight rising smoothly with
// edge strength. Flat regions (weak gradient) are copied through unchanged, so
// the operation is a no-op except on genuine edges.
func edgeGuidedSmooth(src, base *cv.Mat, scale int) *cv.Mat {
	gray := grayscale(src)
	// SobelFloat returns [channel][y*cols+x]; gray is single-channel, so plane 0
	// holds the whole gradient field.
	gx := cv.SobelFloat(gray, 1, 0, 3)[0]
	gy := cv.SobelFloat(gray, 0, 1, 3)[0]

	// Peak gradient magnitude, for an adaptive threshold.
	var maxMag float64
	mag := make([]float64, src.Rows*src.Cols)
	ang := make([]float64, src.Rows*src.Cols)
	for i := 0; i < src.Rows*src.Cols; i++ {
		dx := gx[i]
		dy := gy[i]
		m := math.Hypot(dx, dy)
		mag[i] = m
		ang[i] = math.Atan2(dy, dx)
		if m > maxMag {
			maxMag = m
		}
	}
	dst := base.Clone()
	if maxMag < 1e-9 {
		return dst // perfectly flat image, nothing to smooth
	}
	lo := 0.15 * maxMag
	hi := 0.60 * maxMag
	ch := base.Channels
	fscale := float64(scale)
	for y := 0; y < base.Rows; y++ {
		sy := (float64(y)+0.5)/fscale - 0.5
		syi := clampInt(int(math.Round(sy)), 0, src.Rows-1)
		for x := 0; x < base.Cols; x++ {
			sx := (float64(x)+0.5)/fscale - 0.5
			sxi := clampInt(int(math.Round(sx)), 0, src.Cols-1)
			m := mag[syi*src.Cols+sxi]
			if m <= lo {
				continue
			}
			// Blend weight ramps from 0 at lo to a capped maximum at hi.
			w := (m - lo) / (hi - lo)
			if w > 1 {
				w = 1
			}
			w *= 0.6 // cap so edges stay sharp, only jaggies are shaved
			// Tangent direction = gradient rotated 90° (perpendicular).
			a := ang[syi*src.Cols+sxi]
			tx := -math.Sin(a)
			ty := math.Cos(a)
			di := (y*base.Cols + x) * ch
			for c := 0; c < ch; c++ {
				center := float64(base.Data[di+c])
				fwd := bilinearAt(base, float64(x)+tx, float64(y)+ty, c)
				bwd := bilinearAt(base, float64(x)-tx, float64(y)-ty, c)
				smoothed := 0.5*center + 0.25*fwd + 0.25*bwd
				dst.Data[di+c] = clampByte((1-w)*center + w*smoothed)
			}
		}
	}
	return dst
}

// UpsampleFSRCNN enlarges src by an integer scale (2, 3 or 4) with an
// FSRCNN-STYLE finish: a bicubic base pass followed by a fixed separable
// unsharp-mask sharpening built from hand-chosen kernels. It recovers apparent
// high-frequency detail lost by interpolation, imitating the visual effect of a
// learned upscaler.
//
// This is NOT a trained FSRCNN network and contains no learned weights; it is a
// deterministic fixed-kernel approximation named after the family it evokes. For
// the genuine trained model, use OpenCV's C++ dnn_superres with FSRCNN .pb
// weights. It returns an error for an empty image or an unsupported scale.
func UpsampleFSRCNN(src *cv.Mat, scale int) (*cv.Mat, error) {
	if err := validate(src, scale); err != nil {
		return nil, err
	}
	base, err := UpsampleBicubic(src, scale)
	if err != nil {
		return nil, err
	}
	return unsharpSeparable(base, 1.0, 3, 1.0), nil
}

// unsharpSeparable applies an unsharp mask: dst = src + amount*(src - blur),
// where blur is a separable Gaussian of the given odd kernel size and sigma.
// The result sharpens edges while leaving DC (flat) regions unchanged. It works
// per channel with border replication.
func unsharpSeparable(src *cv.Mat, amount float64, ksize int, sigma float64) *cv.Mat {
	blur := cv.GaussianBlur(src, ksize, sigma)
	ch := src.Channels
	dst := cv.NewMat(src.Rows, src.Cols, ch)
	for i := range dst.Data {
		s := float64(src.Data[i])
		b := float64(blur.Data[i])
		dst.Data[i] = clampByte(s + amount*(s-b))
	}
	return dst
}
