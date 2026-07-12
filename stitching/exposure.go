package stitching

import (
	"image"

	cv "github.com/malcolmston/opencv"
)

// ExposureCompensator removes the brightness and colour differences between the
// overlapping images of a panorama that arise from auto-exposure and vignetting.
// It works in two phases: Feed inspects the images where they overlap and solves
// for a correction gain per image (or per block), and Apply scales an image by
// its learned gain. Compensating before blending stops a visible brightness step
// appearing along the seams.
//
// Implementations are [NoExposureCompensator], [GainCompensator] and
// [BlocksGainCompensator]; one is selected for a [Pipeline] with
// [Pipeline.SetExposureCompensator].
type ExposureCompensator interface {
	// Feed learns the correction from the images. corners[i] is the top-left
	// position of images[i] on the shared canvas and masks[i] marks its valid
	// (positive) pixels; all three slices are index-aligned.
	Feed(corners []image.Point, images []*cv.Mat, masks []*cv.FloatMat)
	// Apply scales images[index] in place by its learned correction. corner is the
	// image's canvas position (used by block-based compensators).
	Apply(index int, corner image.Point, img *cv.Mat)
}

// fullMask returns an all-ones coverage mask of the given size, suitable when an
// image contributes every pixel.
func fullMask(rows, cols int) *cv.FloatMat {
	m := cv.NewFloatMat(rows, cols)
	for i := range m.Data {
		m.Data[i] = 1
	}
	return m
}

// pixelIntensity returns the mean channel value of pixel p (a flat pixel index,
// p = y*cols+x) of img.
func pixelIntensity(img *cv.Mat, p int) float64 {
	base := p * img.Channels
	var s float64
	for c := 0; c < img.Channels; c++ {
		s += float64(img.Data[base+c])
	}
	return s / float64(img.Channels)
}

// applyGain scales every sample of img by g and clamps to the byte range.
func applyGain(img *cv.Mat, g float64) {
	for i := range img.Data {
		img.Data[i] = clampUint8(float64(img.Data[i])*g + 0.5)
	}
}

// overlapStats holds the pixel count and per-image mean intensities over the
// region where two images overlap.
type overlapStats struct {
	n        float64
	meanA    float64
	meanB    float64
	sumA     float64
	sumB     float64
	haveMean bool
}

// pairOverlap accumulates the overlap pixel count and intensity sums between
// image a (at corner ca) and image b (at corner cb), considering only pixels
// valid in both masks.
func pairOverlap(a, b *cv.Mat, ma, mb *cv.FloatMat, ca, cb image.Point) overlapStats {
	x0 := maxInt(ca.X, cb.X)
	y0 := maxInt(ca.Y, cb.Y)
	x1 := minInt(ca.X+a.Cols, cb.X+b.Cols)
	y1 := minInt(ca.Y+a.Rows, cb.Y+b.Rows)
	var st overlapStats
	for gy := y0; gy < y1; gy++ {
		for gx := x0; gx < x1; gx++ {
			pa := (gy-ca.Y)*a.Cols + (gx - ca.X)
			pb := (gy-cb.Y)*b.Cols + (gx - cb.X)
			if ma.Data[pa] <= 0 || mb.Data[pb] <= 0 {
				continue
			}
			st.n++
			st.sumA += pixelIntensity(a, pa)
			st.sumB += pixelIntensity(b, pb)
		}
	}
	if st.n > 0 {
		st.meanA = st.sumA / st.n
		st.meanB = st.sumB / st.n
		st.haveMean = true
	}
	return st
}

// NoExposureCompensator is the identity [ExposureCompensator]; it learns nothing
// and applies no correction. Use it to disable exposure compensation.
type NoExposureCompensator struct{}

// Feed does nothing.
func (NoExposureCompensator) Feed([]image.Point, []*cv.Mat, []*cv.FloatMat) {}

// Apply does nothing.
func (NoExposureCompensator) Apply(int, image.Point, *cv.Mat) {}

// GainCompensator estimates one multiplicative gain per image by minimising the
// intensity mismatch between overlapping images, following Brown and Lowe's
// gain-compensation model: the total squared difference of overlap intensities
// is minimised subject to a prior that keeps the gains near one. The prior
// strength is set by the ratio of SigmaN (the intensity-noise standard
// deviation) to SigmaG (the allowed gain deviation).
type GainCompensator struct {
	// SigmaN is the standard deviation of the intensity error; larger values
	// trust the prior more. Zero selects a sensible default.
	SigmaN float64
	// SigmaG is the standard deviation of the gain from unity; larger values let
	// gains move further from one. Zero selects a sensible default.
	SigmaG float64

	gains []float64
}

// Gains returns the per-image gains learned by the most recent Feed. The slice
// is nil before Feed is called.
func (gc *GainCompensator) Gains() []float64 { return gc.gains }

// Feed solves the linear system that balances the overlap intensities and stores
// one gain per image.
func (gc *GainCompensator) Feed(corners []image.Point, images []*cv.Mat, masks []*cv.FloatMat) {
	n := len(images)
	gc.gains = make([]float64, n)
	for i := range gc.gains {
		gc.gains[i] = 1
	}
	if n == 0 {
		return
	}
	sigmaN := gc.SigmaN
	if sigmaN <= 0 {
		sigmaN = 10
	}
	sigmaG := gc.SigmaG
	if sigmaG <= 0 {
		sigmaG = 0.1
	}
	alpha := 1 / (sigmaN * sigmaN)
	beta := 1 / (sigmaG * sigmaG)

	a := make([][]float64, n)
	for i := range a {
		a[i] = make([]float64, n)
	}
	b := make([]float64, n)
	for i := 0; i < n; i++ {
		a[i][i] += beta
		b[i] += beta
		for j := i + 1; j < n; j++ {
			st := pairOverlap(images[i], images[j], masks[i], masks[j], corners[i], corners[j])
			if !st.haveMean {
				continue
			}
			iij := st.meanA
			iji := st.meanB
			a[i][i] += alpha * st.n * iij * iij
			a[j][j] += alpha * st.n * iji * iji
			a[i][j] -= alpha * st.n * iij * iji
			a[j][i] -= alpha * st.n * iij * iji
		}
	}
	if sol, ok := solveDense(a, b); ok {
		gc.gains = sol
	}
}

// Apply scales the image by its learned gain.
func (gc *GainCompensator) Apply(index int, _ image.Point, img *cv.Mat) {
	if index < 0 || index >= len(gc.gains) {
		return
	}
	applyGain(img, gc.gains[index])
}

// BlocksGainCompensator estimates a spatially-varying gain field per image by
// dividing every image into a grid of blocks and solving for one gain per block.
// Overlap-intensity data terms tie together blocks of different images that cover
// the same ground, and a smoothness prior (Lambda) ties neighbouring blocks of
// the same image together, so vignetting and gradual exposure drift within a
// single frame are corrected, not just a global offset. Gains are interpolated
// bilinearly between block centres when applied.
type BlocksGainCompensator struct {
	// BlockWidth and BlockHeight set the block grid resolution in pixels. Zero
	// selects a 32-pixel default.
	BlockWidth  int
	BlockHeight int
	// Lambda is the smoothness weight between neighbouring blocks of one image.
	// Zero selects a sensible default.
	Lambda float64
	// SigmaN and SigmaG configure the intensity/gain priors as in
	// [GainCompensator].
	SigmaN float64
	SigmaG float64

	blockGains [][]float64 // per image, row-major nby×nbx gains
	nbx        []int
	nby        []int
	bw, bh     int
}

// Feed solves the block-level system and stores a gain grid per image.
func (bc *BlocksGainCompensator) Feed(corners []image.Point, images []*cv.Mat, masks []*cv.FloatMat) {
	n := len(images)
	bc.bw = bc.BlockWidth
	if bc.bw <= 0 {
		bc.bw = 32
	}
	bc.bh = bc.BlockHeight
	if bc.bh <= 0 {
		bc.bh = 32
	}
	lambda := bc.Lambda
	if lambda <= 0 {
		lambda = 5
	}
	sigmaN := bc.SigmaN
	if sigmaN <= 0 {
		sigmaN = 10
	}
	sigmaG := bc.SigmaG
	if sigmaG <= 0 {
		sigmaG = 0.1
	}
	alpha := 1 / (sigmaN * sigmaN)
	beta := 1 / (sigmaG * sigmaG)

	bc.nbx = make([]int, n)
	bc.nby = make([]int, n)
	offsets := make([]int, n+1) // prefix sum of node counts per image
	for i := 0; i < n; i++ {
		bc.nbx[i] = (images[i].Cols + bc.bw - 1) / bc.bw
		bc.nby[i] = (images[i].Rows + bc.bh - 1) / bc.bh
		offsets[i+1] = offsets[i] + bc.nbx[i]*bc.nby[i]
	}
	total := offsets[n]
	bc.blockGains = make([][]float64, n)
	for i := 0; i < n; i++ {
		bc.blockGains[i] = make([]float64, bc.nbx[i]*bc.nby[i])
		for k := range bc.blockGains[i] {
			bc.blockGains[i][k] = 1
		}
	}
	if total == 0 {
		return
	}

	a := make([][]float64, total)
	for i := range a {
		a[i] = make([]float64, total)
	}
	b := make([]float64, total)

	blockOf := func(img int, lx, ly int) int {
		bx := lx / bc.bw
		by := ly / bc.bh
		return offsets[img] + by*bc.nbx[img] + bx
	}

	// Prior: every node's gain is pulled toward one.
	for k := 0; k < total; k++ {
		a[k][k] += beta
		b[k] += beta
	}

	// Data terms between overlapping blocks of different images.
	type acc struct{ n, sa, sb float64 }
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			ca, cb := corners[i], corners[j]
			ai, aj := images[i], images[j]
			mi, mj := masks[i], masks[j]
			x0 := maxInt(ca.X, cb.X)
			y0 := maxInt(ca.Y, cb.Y)
			x1 := minInt(ca.X+ai.Cols, cb.X+aj.Cols)
			y1 := minInt(ca.Y+ai.Rows, cb.Y+aj.Rows)
			pairAcc := map[[2]int]*acc{}
			for gy := y0; gy < y1; gy++ {
				for gx := x0; gx < x1; gx++ {
					lax, lay := gx-ca.X, gy-ca.Y
					lbx, lby := gx-cb.X, gy-cb.Y
					pa := lay*ai.Cols + lax
					pb := lby*aj.Cols + lbx
					if mi.Data[pa] <= 0 || mj.Data[pb] <= 0 {
						continue
					}
					na := blockOf(i, lax, lay)
					nb := blockOf(j, lbx, lby)
					key := [2]int{na, nb}
					e := pairAcc[key]
					if e == nil {
						e = &acc{}
						pairAcc[key] = e
					}
					e.n++
					e.sa += pixelIntensity(ai, pa)
					e.sb += pixelIntensity(aj, pb)
				}
			}
			for key, e := range pairAcc {
				na, nb := key[0], key[1]
				iij := e.sa / e.n
				iji := e.sb / e.n
				a[na][na] += alpha * e.n * iij * iij
				a[nb][nb] += alpha * e.n * iji * iji
				a[na][nb] -= alpha * e.n * iij * iji
				a[nb][na] -= alpha * e.n * iij * iji
			}
		}
	}

	// Smoothness terms between neighbouring blocks within each image.
	for i := 0; i < n; i++ {
		for by := 0; by < bc.nby[i]; by++ {
			for bx := 0; bx < bc.nbx[i]; bx++ {
				cur := offsets[i] + by*bc.nbx[i] + bx
				if bx+1 < bc.nbx[i] {
					right := cur + 1
					addSmooth(a, cur, right, lambda)
				}
				if by+1 < bc.nby[i] {
					down := offsets[i] + (by+1)*bc.nbx[i] + bx
					addSmooth(a, cur, down, lambda)
				}
			}
		}
	}

	sol, ok := solveDense(a, b)
	if !ok {
		return
	}
	for i := 0; i < n; i++ {
		copy(bc.blockGains[i], sol[offsets[i]:offsets[i+1]])
	}
}

// addSmooth adds a symmetric smoothness coupling of weight w between nodes p and
// q of the least-squares matrix.
func addSmooth(a [][]float64, p, q int, w float64) {
	a[p][p] += w
	a[q][q] += w
	a[p][q] -= w
	a[q][p] -= w
}

// Apply multiplies each pixel of the image by its gain, bilinearly interpolated
// from the surrounding block-centre gains.
func (bc *BlocksGainCompensator) Apply(index int, _ image.Point, img *cv.Mat) {
	if index < 0 || index >= len(bc.blockGains) {
		return
	}
	gains := bc.blockGains[index]
	nbx, nby := bc.nbx[index], bc.nby[index]
	for y := 0; y < img.Rows; y++ {
		for x := 0; x < img.Cols; x++ {
			g := bc.sampleBlockGain(gains, nbx, nby, x, y)
			base := (y*img.Cols + x) * img.Channels
			for c := 0; c < img.Channels; c++ {
				img.Data[base+c] = clampUint8(float64(img.Data[base+c])*g + 0.5)
			}
		}
	}
}

// sampleBlockGain bilinearly interpolates the block-gain grid at pixel (x, y),
// treating each gain as located at its block centre.
func (bc *BlocksGainCompensator) sampleBlockGain(gains []float64, nbx, nby, x, y int) float64 {
	fx := (float64(x) - float64(bc.bw)/2) / float64(bc.bw)
	fy := (float64(y) - float64(bc.bh)/2) / float64(bc.bh)
	bx0 := int(fx)
	by0 := int(fy)
	if fx < 0 {
		bx0 = 0
	}
	if fy < 0 {
		by0 = 0
	}
	bx1 := bx0 + 1
	by1 := by0 + 1
	tx := fx - float64(bx0)
	ty := fy - float64(by0)
	if tx < 0 {
		tx = 0
	}
	if ty < 0 {
		ty = 0
	}
	clampx := func(v int) int {
		if v < 0 {
			return 0
		}
		if v >= nbx {
			return nbx - 1
		}
		return v
	}
	clampy := func(v int) int {
		if v < 0 {
			return 0
		}
		if v >= nby {
			return nby - 1
		}
		return v
	}
	g00 := gains[clampy(by0)*nbx+clampx(bx0)]
	g01 := gains[clampy(by0)*nbx+clampx(bx1)]
	g10 := gains[clampy(by1)*nbx+clampx(bx0)]
	g11 := gains[clampy(by1)*nbx+clampx(bx1)]
	top := g00*(1-tx) + g01*tx
	bot := g10*(1-tx) + g11*tx
	return top*(1-ty) + bot*ty
}

// minInt returns the smaller of a and b.
func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// maxInt returns the larger of a and b.
func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
