package xphoto

import (
	"math"
	"sort"

	cv "github.com/malcolmston/opencv"
)

// DarkChannelDehazer removes haze/fog from a single RGB image using He, Sun and
// Tang's dark channel prior. Its zero value is not recommended; construct it
// with [NewDarkChannelDehazer], which fills in sensible defaults, then override
// fields as needed.
//
// The prior observes that in haze-free outdoor images most non-sky patches
// contain some very dark pixel in at least one colour channel; where that fails
// (the "dark channel" is bright) the patch is hazy. The dehazer estimates the
// atmospheric light from the haziest pixels, derives a per-pixel transmission
// map from the dark channel, refines it with an edge-aware guided filter, and
// inverts the haze imaging model I = J*t + A*(1-t) to recover the scene radiance
// J. The result has markedly higher contrast than the hazy input.
type DarkChannelDehazer struct {
	// PatchRadius is the half-size of the local window used for the dark
	// channel and the raw transmission estimate. Values <= 0 default to 3.
	PatchRadius int
	// Omega in (0,1] is the fraction of haze kept, which leaves a little haze
	// for distant objects so the result still looks natural. Values outside
	// (0,1] default to 0.95.
	Omega float64
	// T0 is the lower bound on transmission, preventing division by near-zero in
	// dense haze. Values <= 0 default to 0.1.
	T0 float64
	// GuidedRadius is the window half-size of the guided-filter refinement of
	// the transmission map. Values <= 0 default to 4*PatchRadius.
	GuidedRadius int
	// GuidedEps is the guided-filter regularisation (in normalised [0,1]^2
	// units); larger values smooth more. Values <= 0 default to 1e-3.
	GuidedEps float64
}

// NewDarkChannelDehazer returns a DarkChannelDehazer with the parameters
// recommended by He et al.: patch radius 3, omega 0.95, transmission floor 0.1,
// a guided-filter radius of 12 and epsilon 1e-3.
func NewDarkChannelDehazer() *DarkChannelDehazer {
	return &DarkChannelDehazer{PatchRadius: 3, Omega: 0.95, T0: 0.1, GuidedRadius: 12, GuidedEps: 1e-3}
}

// Dehaze removes haze from src with the dark channel prior and default
// parameters. It is a convenience wrapper over [DarkChannelDehazer.Apply]. src
// must be a three-channel RGB image; the input is not modified.
func Dehaze(src *cv.Mat) *cv.Mat {
	return NewDarkChannelDehazer().Apply(src)
}

// Apply runs the dark channel prior dehazing on src and returns the recovered
// haze-free image. src must be a three-channel RGB image; the input is not
// modified.
func (d *DarkChannelDehazer) Apply(src *cv.Mat) *cv.Mat {
	requireNonEmpty(src, "DarkChannelDehazer.Apply")
	requireChannels(src, 3, "DarkChannelDehazer.Apply")

	patch := d.PatchRadius
	if patch <= 0 {
		patch = 3
	}
	omega := d.Omega
	if omega <= 0 || omega > 1 {
		omega = 0.95
	}
	t0 := d.T0
	if t0 <= 0 {
		t0 = 0.1
	}
	gr := d.GuidedRadius
	if gr <= 0 {
		gr = 4 * patch
	}
	eps := d.GuidedEps
	if eps <= 0 {
		eps = 1e-3
	}

	rows, cols := src.Rows, src.Cols
	n := rows * cols
	// Normalise the image to [0,1] planes.
	img := make([][3]float64, n)
	for i := 0; i < n; i++ {
		img[i] = [3]float64{
			float64(src.Data[i*3+0]) / 255.0,
			float64(src.Data[i*3+1]) / 255.0,
			float64(src.Data[i*3+2]) / 255.0,
		}
	}

	// Dark channel: per pixel the min over channels, then the min over a patch.
	perPixelMin := make([]float64, n)
	for i := 0; i < n; i++ {
		perPixelMin[i] = math.Min(img[i][0], math.Min(img[i][1], img[i][2]))
	}
	dark := minFilter(perPixelMin, rows, cols, patch)

	// Atmospheric light: among the 0.1% haziest pixels (largest dark channel),
	// take the one with the largest input intensity, per channel.
	atmos := estimateAtmosphere(img, dark, rows, cols)

	// Raw transmission: t = 1 - omega * darkChannel(I/A).
	normMin := make([]float64, n)
	for i := 0; i < n; i++ {
		normMin[i] = math.Min(img[i][0]/atmos[0], math.Min(img[i][1]/atmos[1], img[i][2]/atmos[2]))
	}
	darkNorm := minFilter(normMin, rows, cols, patch)
	trans := make([]float64, n)
	for i := 0; i < n; i++ {
		trans[i] = 1 - omega*darkNorm[i]
	}

	// Edge-aware refinement guided by the grayscale image.
	guide := make([]float64, n)
	for i := 0; i < n; i++ {
		guide[i] = luma(img[i][0], img[i][1], img[i][2])
	}
	trans = guidedFilter(guide, trans, rows, cols, gr, eps)

	// Recover scene radiance: J = (I - A)/max(t,t0) + A.
	dst := cv.NewMat(rows, cols, 3)
	for i := 0; i < n; i++ {
		t := math.Max(trans[i], t0)
		for c := 0; c < 3; c++ {
			j := (img[i][c]-atmos[c])/t + atmos[c]
			dst.Data[i*3+c] = clampU8(j * 255.0)
		}
	}
	return dst
}

// minFilter returns the local minimum of plane over a (2r+1) square window with
// edge clamping.
func minFilter(plane []float64, rows, cols, r int) []float64 {
	out := make([]float64, rows*cols)
	for y := 0; y < rows; y++ {
		y0, y1 := y-r, y+r
		if y0 < 0 {
			y0 = 0
		}
		if y1 >= rows {
			y1 = rows - 1
		}
		for x := 0; x < cols; x++ {
			x0, x1 := x-r, x+r
			if x0 < 0 {
				x0 = 0
			}
			if x1 >= cols {
				x1 = cols - 1
			}
			m := math.Inf(1)
			for yy := y0; yy <= y1; yy++ {
				row := yy * cols
				for xx := x0; xx <= x1; xx++ {
					if plane[row+xx] < m {
						m = plane[row+xx]
					}
				}
			}
			out[y*cols+x] = m
		}
	}
	return out
}

// estimateAtmosphere returns the per-channel atmospheric light: from the 0.1%
// of pixels with the largest dark-channel value, the one whose input intensity
// (channel sum) is greatest supplies each channel's air-light value.
func estimateAtmosphere(img [][3]float64, dark []float64, rows, cols int) [3]float64 {
	n := rows * cols
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(a, b int) bool { return dark[idx[a]] > dark[idx[b]] })
	count := n / 1000
	if count < 1 {
		count = 1
	}
	var atmos [3]float64
	bestIntensity := -1.0
	for k := 0; k < count; k++ {
		i := idx[k]
		intensity := img[i][0] + img[i][1] + img[i][2]
		if intensity > bestIntensity {
			bestIntensity = intensity
			atmos = img[i]
		}
	}
	for c := 0; c < 3; c++ {
		if atmos[c] < 1e-4 {
			atmos[c] = 1e-4
		}
	}
	return atmos
}

// guidedFilter applies He et al.'s guided filter to p with grayscale guide,
// window radius r and regularisation eps, returning an edge-preserving smoothed
// version of p whose edges follow guide.
func guidedFilter(guide, p []float64, rows, cols, r int, eps float64) []float64 {
	n := rows * cols
	ip := make([]float64, n)
	ii := make([]float64, n)
	for i := 0; i < n; i++ {
		ip[i] = guide[i] * p[i]
		ii[i] = guide[i] * guide[i]
	}
	meanI := boxFilter(guide, rows, cols, r)
	meanP := boxFilter(p, rows, cols, r)
	meanIp := boxFilter(ip, rows, cols, r)
	meanII := boxFilter(ii, rows, cols, r)

	a := make([]float64, n)
	b := make([]float64, n)
	for i := 0; i < n; i++ {
		varI := meanII[i] - meanI[i]*meanI[i]
		covIp := meanIp[i] - meanI[i]*meanP[i]
		a[i] = covIp / (varI + eps)
		b[i] = meanP[i] - a[i]*meanI[i]
	}
	meanA := boxFilter(a, rows, cols, r)
	meanB := boxFilter(b, rows, cols, r)
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		out[i] = meanA[i]*guide[i] + meanB[i]
	}
	return out
}

// boxFilter returns the local mean of plane over a (2r+1) square window with
// edge clamping, computed with a summed-area table for O(1) per-pixel cost.
func boxFilter(plane []float64, rows, cols, r int) []float64 {
	// Integral image with a zero-padded first row/column.
	sat := make([]float64, (rows+1)*(cols+1))
	stride := cols + 1
	for y := 0; y < rows; y++ {
		var rowSum float64
		for x := 0; x < cols; x++ {
			rowSum += plane[y*cols+x]
			sat[(y+1)*stride+(x+1)] = sat[y*stride+(x+1)] + rowSum
		}
	}
	out := make([]float64, rows*cols)
	for y := 0; y < rows; y++ {
		y0 := y - r
		y1 := y + r
		if y0 < 0 {
			y0 = 0
		}
		if y1 >= rows {
			y1 = rows - 1
		}
		for x := 0; x < cols; x++ {
			x0 := x - r
			x1 := x + r
			if x0 < 0 {
				x0 = 0
			}
			if x1 >= cols {
				x1 = cols - 1
			}
			area := float64((y1 - y0 + 1) * (x1 - x0 + 1))
			s := sat[(y1+1)*stride+(x1+1)] - sat[y0*stride+(x1+1)] - sat[(y1+1)*stride+x0] + sat[y0*stride+x0]
			out[y*cols+x] = s / area
		}
	}
	return out
}
