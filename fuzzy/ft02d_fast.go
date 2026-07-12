package fuzzy

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
)

// flSeparable computes the degree-0 F-transform reconstruction of img with the
// given basis function and radius using a separable two-pass algorithm, and
// returns the real-valued (un-rounded) reconstruction in row-major, channel-
// interleaved order.
//
// Because the fuzzy-partition kernel is the outer product of a 1-D membership
// vector with itself, both the forward components and the inverse reconstruction
// factor into independent horizontal and vertical passes. This lowers the cost
// from O(rows*cols*radius^2) for the dense [FT02DProcess] to O(rows*cols*radius),
// while producing the same partition-of-unity-normalised result. It is the engine
// behind the "FL" (fast, linear-time) process variants and assumes no validity
// mask (every pixel participates), matching OpenCV's fast ft::FT02D_FL_process.
func flSeparable(img *cv.Mat, function BasisFunction, radius int) []float64 {
	rows, cols, ch := img.Rows, img.Cols, img.Channels
	v := basisVector(function, radius)
	bn, an := partitionCounts(rows, cols, radius)
	kside := 2*radius + 1

	// --- forward, horizontal pass -----------------------------------------
	// H[(iy*an+o)*ch+cc] = sum over the kernel columns of v[kx]*img, and
	// sx[o] = sum of v[kx] over in-bounds columns (the horizontal denominator).
	h := make([]float64, rows*an*ch)
	sx := make([]float64, an)
	for o := 0; o < an; o++ {
		originX := o*radius - radius
		for kx := 0; kx < kside; kx++ {
			ix := originX + kx
			if ix < 0 || ix >= cols {
				continue
			}
			sx[o] += v[kx]
		}
		for iy := 0; iy < rows; iy++ {
			for kx := 0; kx < kside; kx++ {
				ix := originX + kx
				if ix < 0 || ix >= cols {
					continue
				}
				w := v[kx]
				base := (iy*cols + ix) * ch
				hbase := (iy*an + o) * ch
				for cc := 0; cc < ch; cc++ {
					h[hbase+cc] += w * float64(img.Data[base+cc])
				}
			}
		}
	}

	// --- forward, vertical pass -> node components -------------------------
	comp := make([]float64, bn*an*ch)
	for i := 0; i < bn; i++ {
		originY := i*radius - radius
		var sy float64
		for ky := 0; ky < kside; ky++ {
			iy := originY + ky
			if iy < 0 || iy >= rows {
				continue
			}
			sy += v[ky]
		}
		for o := 0; o < an; o++ {
			den := sy * sx[o]
			cbase := (i*an + o) * ch
			for ky := 0; ky < kside; ky++ {
				iy := originY + ky
				if iy < 0 || iy >= rows {
					continue
				}
				w := v[ky]
				hbase := (iy*an + o) * ch
				for cc := 0; cc < ch; cc++ {
					comp[cbase+cc] += w * h[hbase+cc]
				}
			}
			if den > 0 {
				for cc := 0; cc < ch; cc++ {
					comp[cbase+cc] /= den
				}
			}
		}
	}

	// --- inverse, horizontal spread ---------------------------------------
	// g[(i*cols+ix)*ch+cc] = sum over nodes o of vx(ix)*comp, and
	// wx[ix] = sum of vx(ix) over covering nodes (horizontal weight).
	g := make([]float64, bn*cols*ch)
	wx := make([]float64, cols)
	for o := 0; o < an; o++ {
		center := o * radius
		for d := -radius; d <= radius; d++ {
			ix := center + d
			if ix < 0 || ix >= cols {
				continue
			}
			w := v[d+radius]
			if w == 0 {
				continue
			}
			wx[ix] += w // total horizontal partition weight reaching pixel ix.
		}
	}
	for i := 0; i < bn; i++ {
		for o := 0; o < an; o++ {
			center := o * radius
			cbase := (i*an + o) * ch
			for d := -radius; d <= radius; d++ {
				ix := center + d
				if ix < 0 || ix >= cols {
					continue
				}
				w := v[d+radius]
				if w == 0 {
					continue
				}
				gbase := (i*cols + ix) * ch
				for cc := 0; cc < ch; cc++ {
					g[gbase+cc] += w * comp[cbase+cc]
				}
			}
		}
	}

	// --- inverse, vertical spread -> reconstruction -----------------------
	wy := make([]float64, rows)
	for i := 0; i < bn; i++ {
		center := i * radius
		for d := -radius; d <= radius; d++ {
			iy := center + d
			if iy < 0 || iy >= rows {
				continue
			}
			wy[iy] += v[d+radius]
		}
	}
	out := make([]float64, rows*cols*ch)
	for i := 0; i < bn; i++ {
		center := i * radius
		for d := -radius; d <= radius; d++ {
			iy := center + d
			if iy < 0 || iy >= rows {
				continue
			}
			w := v[d+radius]
			if w == 0 {
				continue
			}
			for ix := 0; ix < cols; ix++ {
				norm := wy[iy] * wx[ix]
				if norm <= 0 {
					continue
				}
				gbase := (i*cols + ix) * ch
				obase := (iy*cols + ix) * ch
				for cc := 0; cc < ch; cc++ {
					out[obase+cc] += w * g[gbase+cc] / norm
				}
			}
		}
	}
	return out
}

// FT02DFLProcess smooths img with the degree-0 F-transform using the fast,
// separable linear-time algorithm, mirroring OpenCV's ft::FT02D_FL_process. It
// produces the same partition-of-unity reconstruction as [FT02DProcess] with a
// nil mask but at O(rows*cols*radius) rather than O(rows*cols*radius^2) cost by
// exploiting the separability of the fuzzy-partition kernel. It works on
// grayscale or multi-channel (e.g. RGB) images and returns a fresh [cv.Mat] with
// values rounded and clamped to the 8-bit range; the input is not modified.
func FT02DFLProcess(img *cv.Mat, function BasisFunction, radius int) *cv.Mat {
	if img == nil || img.Empty() {
		panic("fuzzy: FT02DFLProcess given an empty image")
	}
	if radius < 1 {
		panic(fmt.Sprintf("fuzzy: FT02DFLProcess radius must be >= 1, got %d", radius))
	}
	rec := flSeparable(img, function, radius)
	out := cv.NewMat(img.Rows, img.Cols, img.Channels)
	for i := range out.Data {
		out.Data[i] = clampByte(rec[i])
	}
	return out
}

// FT02DFLProcessFloat is the real-valued counterpart of [FT02DFLProcess],
// mirroring OpenCV's ft::FT02D_FL_process_float. It runs the same fast separable
// reconstruction but returns the un-rounded result as a single-channel
// [cv.FloatMat], preserving sub-integer precision for downstream numeric use
// (error analysis, further filtering). Because [cv.FloatMat] is single-channel,
// img must be single-channel; use [FT02DFLProcess] for colour images.
func FT02DFLProcessFloat(img *cv.Mat, function BasisFunction, radius int) *cv.FloatMat {
	if img == nil || img.Empty() {
		panic("fuzzy: FT02DFLProcessFloat given an empty image")
	}
	if img.Channels != 1 {
		panic(fmt.Sprintf("fuzzy: FT02DFLProcessFloat requires a single-channel image, got %d channels", img.Channels))
	}
	if radius < 1 {
		panic(fmt.Sprintf("fuzzy: FT02DFLProcessFloat radius must be >= 1, got %d", radius))
	}
	rec := flSeparable(img, function, radius)
	out := cv.NewFloatMat(img.Rows, img.Cols)
	copy(out.Data, rec)
	return out
}
