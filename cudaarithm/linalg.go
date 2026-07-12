package cudaarithm

import (
	"fmt"
	"math"

	cv "github.com/malcolmston/opencv"
)

// Gemm computes the saturating general matrix product alpha*A*B + beta*C and
// returns it as a single-channel GpuMat, mirroring cv::cuda::gemm (without the
// transpose flags). A must be m×k, B must be k×n and, when non-nil, C must be
// m×n; all operands must be single-channel. Pass a nil c to omit the beta*C
// term. Products can easily exceed a byte, so results are rounded and saturated
// into [0,255]; keep operands small for exact results.
func Gemm(a, b *GpuMat, alpha float64, c *GpuMat, beta float64, _ ...*Stream) *GpuMat {
	requireChannels(a, 1, "Gemm")
	requireChannels(b, 1, "Gemm")
	m, k := a.mat.Rows, a.mat.Cols
	k2, n := b.mat.Rows, b.mat.Cols
	if k != k2 {
		panic(fmt.Sprintf("cudaarithm: Gemm inner dimensions disagree: A is %dx%d, B is %dx%d", m, k, k2, n))
	}
	if c != nil {
		requireChannels(c, 1, "Gemm")
		if c.mat.Rows != m || c.mat.Cols != n {
			panic(fmt.Sprintf("cudaarithm: Gemm C must be %dx%d, got %dx%d", m, n, c.mat.Rows, c.mat.Cols))
		}
	}
	out := cv.NewMat(m, n, 1)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var acc float64
			for p := 0; p < k; p++ {
				acc += float64(a.mat.Data[i*k+p]) * float64(b.mat.Data[p*n+j])
			}
			v := alpha * acc
			if c != nil {
				v += beta * float64(c.mat.Data[i*n+j])
			}
			out.Data[i*n+j] = roundToUint8(v)
		}
	}
	return wrap(out)
}

// ComplexMat is a dense, row-major complex matrix used by [DFT], [IDFT] and
// [MulSpectrums]. It exists because a [GpuMat]'s underlying [cv.Mat] stores only
// 8-bit samples and cannot represent the range of Fourier coefficients; this is
// the one place the CUDA-mirror API necessarily diverges from operating on a
// GpuMat. Re and Im each have length Rows*Cols.
type ComplexMat struct {
	Rows int
	Cols int
	Re   []float64
	Im   []float64
}

// NewComplexMat allocates a zero-filled ComplexMat.
func NewComplexMat(rows, cols int) *ComplexMat {
	return &ComplexMat{Rows: rows, Cols: cols, Re: make([]float64, rows*cols), Im: make([]float64, rows*cols)}
}

// At returns the real and imaginary parts at row y, column x.
func (c *ComplexMat) At(y, x int) (re, im float64) {
	i := y*c.Cols + x
	return c.Re[i], c.Im[i]
}

// DFT computes the forward two-dimensional discrete Fourier transform of a
// single-channel GpuMat and returns the complex spectrum, mirroring
// cv::cuda::dft. The transform is a genuine, exact DFT (a direct evaluation of
// the definition, not a fast-transform approximation), so it is O((rows*cols)^2)
// and intended for modest sizes. Real input is taken from the samples; the
// imaginary part starts at zero. It panics unless src is single-channel.
func DFT(src *GpuMat, _ ...*Stream) *ComplexMat {
	requireChannels(src, 1, "DFT")
	rows, cols := src.mat.Rows, src.mat.Cols
	in := NewComplexMat(rows, cols)
	for i, s := range src.mat.Data {
		in.Re[i] = float64(s)
	}
	return dft2D(in, false)
}

// IDFT computes the inverse two-dimensional DFT of a spectrum and returns the
// real part as a rounded, saturated single-channel GpuMat, mirroring
// cv::cuda::dft with the inverse flag. The inverse is normalised by 1/(rows*cols)
// so that IDFT(DFT(x)) reproduces x (up to rounding). Negative real parts clamp
// to 0 and parts above 255 clamp to 255.
func IDFT(spec *ComplexMat, _ ...*Stream) *GpuMat {
	if spec == nil || spec.Rows <= 0 || spec.Cols <= 0 {
		panic("cudaarithm: IDFT given an empty spectrum")
	}
	res := dft2D(spec, true)
	scale := 1.0 / float64(spec.Rows*spec.Cols)
	out := cv.NewMat(spec.Rows, spec.Cols, 1)
	for i := range res.Re {
		out.Data[i] = roundToUint8(res.Re[i] * scale)
	}
	return wrap(out)
}

// IDFTComplex computes the inverse two-dimensional DFT and returns the full
// complex result (normalised by 1/(rows*cols)) without rounding to bytes. It is
// useful when the inverse transform is an intermediate result rather than a
// displayable image.
func IDFTComplex(spec *ComplexMat, _ ...*Stream) *ComplexMat {
	if spec == nil || spec.Rows <= 0 || spec.Cols <= 0 {
		panic("cudaarithm: IDFTComplex given an empty spectrum")
	}
	res := dft2D(spec, true)
	scale := 1.0 / float64(spec.Rows*spec.Cols)
	for i := range res.Re {
		res.Re[i] *= scale
		res.Im[i] *= scale
	}
	return res
}

// MulSpectrums multiplies two spectra element-wise, mirroring
// cv::cuda::mulSpectrums. When conjB is true the complex conjugate of b is used,
// which corresponds to correlation rather than convolution. The operands must
// share a shape. Combined with [DFT] and [IDFT] this realises the convolution
// theorem: IDFT(MulSpectrums(DFT(a), DFT(b))) is the circular convolution of a
// and b.
func MulSpectrums(a, b *ComplexMat, conjB bool, _ ...*Stream) *ComplexMat {
	if a == nil || b == nil {
		panic("cudaarithm: MulSpectrums given a nil spectrum")
	}
	if a.Rows != b.Rows || a.Cols != b.Cols {
		panic(fmt.Sprintf("cudaarithm: MulSpectrums shape mismatch %dx%d vs %dx%d", a.Rows, a.Cols, b.Rows, b.Cols))
	}
	out := NewComplexMat(a.Rows, a.Cols)
	for i := range a.Re {
		br, bi := b.Re[i], b.Im[i]
		if conjB {
			bi = -bi
		}
		out.Re[i] = a.Re[i]*br - a.Im[i]*bi
		out.Im[i] = a.Re[i]*bi + a.Im[i]*br
	}
	return out
}

// dft2D applies a separable 2-D DFT (over rows then columns) to in. When inverse
// is true the +2πi sign convention is used; the caller applies any 1/N scaling.
func dft2D(in *ComplexMat, inverse bool) *ComplexMat {
	rows, cols := in.Rows, in.Cols
	tmp := &ComplexMat{Rows: rows, Cols: cols, Re: append([]float64(nil), in.Re...), Im: append([]float64(nil), in.Im...)}
	// Transform each row.
	for y := 0; y < rows; y++ {
		base := y * cols
		outRe, outIm := dft1D(tmp.Re[base:base+cols], tmp.Im[base:base+cols], inverse)
		copy(tmp.Re[base:base+cols], outRe)
		copy(tmp.Im[base:base+cols], outIm)
	}
	// Transform each column.
	colRe := make([]float64, rows)
	colIm := make([]float64, rows)
	for x := 0; x < cols; x++ {
		for y := 0; y < rows; y++ {
			colRe[y] = tmp.Re[y*cols+x]
			colIm[y] = tmp.Im[y*cols+x]
		}
		outRe, outIm := dft1D(colRe, colIm, inverse)
		for y := 0; y < rows; y++ {
			tmp.Re[y*cols+x] = outRe[y]
			tmp.Im[y*cols+x] = outIm[y]
		}
	}
	return tmp
}

// dft1D evaluates the discrete Fourier transform of a single complex sequence by
// direct summation. The sign of the exponent flips for the inverse transform; no
// normalisation is applied here.
func dft1D(re, im []float64, inverse bool) (outRe, outIm []float64) {
	n := len(re)
	outRe = make([]float64, n)
	outIm = make([]float64, n)
	sign := -1.0
	if inverse {
		sign = 1.0
	}
	for k := 0; k < n; k++ {
		var sr, si float64
		for t := 0; t < n; t++ {
			ang := sign * 2 * math.Pi * float64(k) * float64(t) / float64(n)
			c := math.Cos(ang)
			s := math.Sin(ang)
			sr += re[t]*c - im[t]*s
			si += re[t]*s + im[t]*c
		}
		outRe[k] = sr
		outIm[k] = si
	}
	return outRe, outIm
}
