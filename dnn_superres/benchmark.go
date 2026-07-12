package dnn_superres

import (
	"fmt"
	"sort"

	cv "github.com/malcolmston/opencv"
)

// NamedUpsampler pairs a human-readable label with an [UpsampleFunc] so that
// [Benchmark] can report per-method scores.
type NamedUpsampler struct {
	// Name identifies the method in benchmark output.
	Name string
	// Func is the upsampler to evaluate.
	Func UpsampleFunc
}

// BenchmarkResult holds the reconstruction-quality scores of one method,
// measured against the original high-resolution reference.
type BenchmarkResult struct {
	// Name is the method label.
	Name string
	// PSNR is the peak signal-to-noise ratio in dB (higher is better).
	PSNR float64
	// SSIM is the mean structural similarity in [-1,1] (higher is better).
	SSIM float64
	// MSE is the mean squared error (lower is better).
	MSE float64
}

// DefaultUpsamplers returns a representative set of the package's arbitrary-scale
// methods for use with [Benchmark]. Every method listed accepts any integer
// scale of 2 or more, so the set can be benchmarked at any factor.
func DefaultUpsamplers() []NamedUpsampler {
	return []NamedUpsampler{
		{"gaussian", UpsampleGaussian},
		{"bspline", UpsampleBSpline},
		{"bicubic", UpsampleScale},
		{"mitchell", UpsampleMitchell},
		{"lanczos3", UpsampleLanczos3},
		{"espcn", UpsampleESPCN},
		{"nedi", UpsampleNEDI},
		{"dcci", UpsampleDCCI},
		{"lapsrn", UpsampleLapSRN},
		{"gradient-profile", UpsampleGradientProfile},
		{"ibp", UpsampleIBP},
	}
}

// boxDownsample shrinks src by an integer factor using non-overlapping box
// averaging, the forward degradation model the benchmark inverts. src is first
// cropped so its dimensions are exact multiples of factor.
func boxDownsample(src *cv.Mat, factor int) *cv.Mat {
	lh, lw := src.Rows/factor, src.Cols/factor
	ch := src.Channels
	lo := cv.NewMat(lh, lw, ch)
	inv := 1.0 / float64(factor*factor)
	for y := 0; y < lh; y++ {
		for x := 0; x < lw; x++ {
			for c := 0; c < ch; c++ {
				var sum float64
				for dy := 0; dy < factor; dy++ {
					for dx := 0; dx < factor; dx++ {
						sum += float64(src.Data[((y*factor+dy)*src.Cols+(x*factor+dx))*ch+c])
					}
				}
				lo.Data[(y*lw+x)*ch+c] = clampByte(sum * inv)
			}
		}
	}
	return lo
}

// Benchmark compares super-resolution methods on a controlled reconstruction
// task. The high-resolution reference hi is box-downsampled by scale to make a
// low-resolution input, each method upscales that input back to the reference
// size, and the result is scored against hi with PSNR, SSIM and MSE. The
// returned slice is sorted best-first (descending PSNR), so results[0] is the
// method that reconstructed hi most faithfully.
//
// If methods is nil, [DefaultUpsamplers] is used. It returns an error for an
// empty reference, a scale below 2, a reference smaller than scale in either
// dimension, or from any method that fails on the input.
func Benchmark(hi *cv.Mat, scale int, methods []NamedUpsampler) ([]BenchmarkResult, error) {
	if hi == nil || hi.Empty() {
		return nil, fmt.Errorf("dnn_superres: Benchmark given an empty image")
	}
	if scale < 2 {
		return nil, fmt.Errorf("dnn_superres: unsupported scale %d (want >= 2)", scale)
	}
	if hi.Rows < scale || hi.Cols < scale {
		return nil, fmt.Errorf("dnn_superres: reference %dx%d too small for scale %d", hi.Rows, hi.Cols, scale)
	}
	if methods == nil {
		methods = DefaultUpsamplers()
	}
	lo := boxDownsample(hi, scale)
	// Reference is hi cropped to the exact reconstructed size.
	ref := hi.Region(0, 0, lo.Rows*scale, lo.Cols*scale)
	results := make([]BenchmarkResult, 0, len(methods))
	for _, m := range methods {
		out, err := m.Func(lo, scale)
		if err != nil {
			return nil, fmt.Errorf("dnn_superres: method %q: %w", m.Name, err)
		}
		if out.Rows != ref.Rows || out.Cols != ref.Cols {
			return nil, fmt.Errorf("dnn_superres: method %q produced %dx%d, want %dx%d",
				m.Name, out.Rows, out.Cols, ref.Rows, ref.Cols)
		}
		psnr, err := PSNR(out, ref)
		if err != nil {
			return nil, err
		}
		ssim, err := SSIM(out, ref)
		if err != nil {
			return nil, err
		}
		mse, err := MSE(out, ref)
		if err != nil {
			return nil, err
		}
		results = append(results, BenchmarkResult{Name: m.Name, PSNR: psnr, SSIM: ssim, MSE: mse})
	}
	sort.SliceStable(results, func(i, j int) bool {
		return results[i].PSNR > results[j].PSNR
	})
	return results, nil
}
