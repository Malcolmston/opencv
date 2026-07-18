package features3

import (
	"math"
	"sort"

	cv "github.com/malcolmston/opencv"
)

// GaussianScaleSpace returns the image blurred at each of the given sigmas as a
// slice of cv.FloatMats, the shared building block of the scale-space blob
// detectors. Colour input is converted to grayscale first. It panics if sigmas
// is empty.
func GaussianScaleSpace(img *cv.Mat, sigmas []float64) []*cv.FloatMat {
	if len(sigmas) == 0 {
		panic("features3: GaussianScaleSpace requires at least one sigma")
	}
	g := features3ToGray(img)
	out := make([]*cv.FloatMat, len(sigmas))
	for i, s := range sigmas {
		b := features3gaussianBlur(g, s)
		fm := cv.NewFloatMat(g.Rows, g.Cols)
		copy(fm.Data, b.Data)
		out[i] = fm
	}
	return out
}

// DifferenceOfGaussians returns the Difference-of-Gaussians of an image: the
// image blurred at sigma1 minus the image blurred at sigma2, as a cv.FloatMat.
// With sigma2 > sigma1 this approximates a scale-normalised Laplacian and is
// positive at bright blobs. Colour input is converted to grayscale first.
func DifferenceOfGaussians(img *cv.Mat, sigma1, sigma2 float64) *cv.FloatMat {
	g := features3ToGray(img)
	b1 := features3gaussianBlur(g, sigma1)
	b2 := features3gaussianBlur(g, sigma2)
	res := cv.NewFloatMat(g.Rows, g.Cols)
	for i := range res.Data {
		res.Data[i] = b1.Data[i] - b2.Data[i]
	}
	return res
}

// LaplacianResponse returns the scale-normalised Laplacian-of-Gaussian response
// sigma^2*(Lxx+Lyy) of an image blurred at the given sigma, as a cv.FloatMat.
// Bright blobs of radius sigma*sqrt(2) produce strong negative responses and
// dark blobs strong positive ones. Colour input is converted to grayscale first.
func LaplacianResponse(img *cv.Mat, sigma float64) *cv.FloatMat {
	g := features3ToGray(img)
	b := features3gaussianBlur(g, sigma)
	return features3laplacian(b, sigma)
}

// features3laplacian computes sigma^2*(Lxx+Lyy) of an already-blurred buffer.
func features3laplacian(b *features3gray, sigma float64) *cv.FloatMat {
	res := cv.NewFloatMat(b.Rows, b.Cols)
	norm := sigma * sigma
	for y := 0; y < b.Rows; y++ {
		for x := 0; x < b.Cols; x++ {
			c := b.at(x, y)
			lxx := b.atClamped(x+1, y) - 2*c + b.atClamped(x-1, y)
			lyy := b.atClamped(x, y+1) - 2*c + b.atClamped(x, y-1)
			res.Data[y*b.Cols+x] = norm * (lxx + lyy)
		}
	}
	return res
}

// HessianDeterminant returns the scale-normalised determinant of the Hessian
// sigma^4*(Lxx*Lyy - Lxy^2) of an image blurred at the given sigma, as a
// cv.FloatMat. Blob-like structures produce strong positive responses. Colour
// input is converted to grayscale first.
func HessianDeterminant(img *cv.Mat, sigma float64) *cv.FloatMat {
	g := features3ToGray(img)
	b := features3gaussianBlur(g, sigma)
	return features3hessianDet(b, sigma)
}

// features3hessianDet computes sigma^4*det(H) of an already-blurred buffer.
func features3hessianDet(b *features3gray, sigma float64) *cv.FloatMat {
	res := cv.NewFloatMat(b.Rows, b.Cols)
	norm := sigma * sigma * sigma * sigma
	for y := 0; y < b.Rows; y++ {
		for x := 0; x < b.Cols; x++ {
			c := b.at(x, y)
			lxx := b.atClamped(x+1, y) - 2*c + b.atClamped(x-1, y)
			lyy := b.atClamped(x, y+1) - 2*c + b.atClamped(x, y-1)
			lxy := (b.atClamped(x+1, y+1) - b.atClamped(x-1, y+1) -
				b.atClamped(x+1, y-1) + b.atClamped(x-1, y-1)) / 4
			res.Data[y*b.Cols+x] = norm * (lxx*lyy - lxy*lxy)
		}
	}
	return res
}

// features3logSigmas builds a geometric sequence of numScales sigmas from
// minSigma to maxSigma inclusive.
func features3logSigmas(minSigma, maxSigma float64, numScales int) []float64 {
	if numScales < 1 {
		numScales = 1
	}
	sig := make([]float64, numScales)
	if numScales == 1 {
		sig[0] = minSigma
		return sig
	}
	k := math.Pow(maxSigma/minSigma, 1.0/float64(numScales-1))
	for i := 0; i < numScales; i++ {
		sig[i] = minSigma * math.Pow(k, float64(i))
	}
	return sig
}

// features3blobExtrema scans a scale-space stack of signed responses and returns
// blobs at pixels that are a strict extremum (maximum or minimum) of the signed
// response over the 3×3×3 spatial/scale neighbourhood and whose magnitude is at
// least threshold. The Blob.Response is the response magnitude.
func features3blobExtrema(stack []*cv.FloatMat, sigmas []float64, threshold float64) []Blob {
	if len(stack) == 0 {
		return nil
	}
	rows, cols := stack[0].Rows, stack[0].Cols
	var blobs []Blob
	for s := 0; s < len(stack); s++ {
		for y := 1; y < rows-1; y++ {
			for x := 1; x < cols-1; x++ {
				v := stack[s].Data[y*cols+x]
				if math.Abs(v) < threshold {
					continue
				}
				isMax := true
				isMin := true
				for ds := -1; ds <= 1 && (isMax || isMin); ds++ {
					si := s + ds
					if si < 0 || si >= len(stack) {
						continue
					}
					for dy := -1; dy <= 1; dy++ {
						for dx := -1; dx <= 1; dx++ {
							if ds == 0 && dx == 0 && dy == 0 {
								continue
							}
							nv := stack[si].Data[(y+dy)*cols+(x+dx)]
							if nv >= v {
								isMax = false
							}
							if nv <= v {
								isMin = false
							}
						}
					}
				}
				if isMax || isMin {
					blobs = append(blobs, Blob{
						X: float64(x), Y: float64(y), Sigma: sigmas[s], Response: math.Abs(v),
					})
				}
			}
		}
	}
	return blobs
}

// LoGBlobs detects blobs with the scale-normalised Laplacian-of-Gaussian. It
// builds a stack of [LaplacianResponse] over numScales geometric sigmas from
// minSigma to maxSigma, then reports pixels that are a strict 3×3×3 extremum of
// the signed response with magnitude at least threshold. Both bright and dark
// blobs are found. Results are sorted by descending response. Colour input is
// converted to grayscale first.
func LoGBlobs(img *cv.Mat, minSigma, maxSigma float64, numScales int, threshold float64) []Blob {
	g := features3ToGray(img)
	sigmas := features3logSigmas(minSigma, maxSigma, numScales)
	stack := make([]*cv.FloatMat, len(sigmas))
	for i, s := range sigmas {
		b := features3gaussianBlur(g, s)
		stack[i] = features3laplacian(b, s)
	}
	blobs := features3blobExtrema(stack, sigmas, threshold)
	features3sortBlobs(blobs)
	return blobs
}

// DoGBlobs detects blobs with the Difference-of-Gaussians approximation to the
// Laplacian. It builds a Gaussian pyramid over numScales+1 geometric sigmas from
// minSigma to maxSigma, forms the numScales successive differences and reports
// pixels that are a strict 3×3×3 extremum of the difference with magnitude at
// least threshold. Results are sorted by descending response. Colour input is
// converted to grayscale first.
func DoGBlobs(img *cv.Mat, minSigma, maxSigma float64, numScales int, threshold float64) []Blob {
	g := features3ToGray(img)
	gsig := features3logSigmas(minSigma, maxSigma, numScales+1)
	blur := make([]*features3gray, len(gsig))
	for i, s := range gsig {
		blur[i] = features3gaussianBlur(g, s)
	}
	stack := make([]*cv.FloatMat, numScales)
	dogSig := make([]float64, numScales)
	for i := 0; i < numScales; i++ {
		fm := cv.NewFloatMat(g.Rows, g.Cols)
		for j := range fm.Data {
			// Bright blob -> positive DoG: smaller-sigma blur minus larger-sigma.
			fm.Data[j] = blur[i].Data[j] - blur[i+1].Data[j]
		}
		stack[i] = fm
		dogSig[i] = gsig[i]
	}
	blobs := features3blobExtrema(stack, dogSig, threshold)
	features3sortBlobs(blobs)
	return blobs
}

// DoHBlobs detects blobs with the scale-normalised determinant of the Hessian.
// It builds a stack of [HessianDeterminant] over numScales geometric sigmas from
// minSigma to maxSigma and reports pixels that are a strict 3×3×3 maximum of the
// (positive) determinant with value at least threshold. Results are sorted by
// descending response. Colour input is converted to grayscale first.
func DoHBlobs(img *cv.Mat, minSigma, maxSigma float64, numScales int, threshold float64) []Blob {
	g := features3ToGray(img)
	sigmas := features3logSigmas(minSigma, maxSigma, numScales)
	stack := make([]*cv.FloatMat, len(sigmas))
	for i, s := range sigmas {
		b := features3gaussianBlur(g, s)
		stack[i] = features3hessianDet(b, s)
	}
	// Determinant of Hessian is positive for blobs; detect maxima only.
	rows, cols := g.Rows, g.Cols
	var blobs []Blob
	for s := 0; s < len(stack); s++ {
		for y := 1; y < rows-1; y++ {
			for x := 1; x < cols-1; x++ {
				v := stack[s].Data[y*cols+x]
				if v < threshold {
					continue
				}
				isMax := true
				for ds := -1; ds <= 1 && isMax; ds++ {
					si := s + ds
					if si < 0 || si >= len(stack) {
						continue
					}
					for dy := -1; dy <= 1 && isMax; dy++ {
						for dx := -1; dx <= 1; dx++ {
							if ds == 0 && dx == 0 && dy == 0 {
								continue
							}
							if stack[si].Data[(y+dy)*cols+(x+dx)] >= v {
								isMax = false
								break
							}
						}
					}
				}
				if isMax {
					blobs = append(blobs, Blob{X: float64(x), Y: float64(y), Sigma: sigmas[s], Response: v})
				}
			}
		}
	}
	features3sortBlobs(blobs)
	return blobs
}

// features3sortBlobs sorts blobs by descending response, breaking ties
// deterministically by position and scale.
func features3sortBlobs(blobs []Blob) {
	sort.SliceStable(blobs, func(i, j int) bool {
		if blobs[i].Response != blobs[j].Response {
			return blobs[i].Response > blobs[j].Response
		}
		if blobs[i].Y != blobs[j].Y {
			return blobs[i].Y < blobs[j].Y
		}
		if blobs[i].X != blobs[j].X {
			return blobs[i].X < blobs[j].X
		}
		return blobs[i].Sigma < blobs[j].Sigma
	})
}
