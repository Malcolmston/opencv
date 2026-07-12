package features2d

import (
	"math"
	"sort"

	cv "github.com/malcolmston/opencv"
)

// SIFT parameters and fixed constants, following Lowe's algorithm and OpenCV's
// cv::SIFT.
const (
	siftInitSigma    = 0.5 // assumed blur of the input image
	siftImgBorder    = 5   // ignore extrema within this many pixels of the edge
	siftMaxInterp    = 5   // max sub-pixel interpolation steps
	siftOriHistBins  = 36
	siftOriSigFctr   = 1.5
	siftOriRadius    = 3 * siftOriSigFctr
	siftOriPeakRatio = 0.8
	siftDescrWidth   = 4 // 4×4 spatial grid
	siftDescrHistBin = 8 // 8 orientation bins per cell
	siftDescrSclFctr = 3.0
	siftDescrMagThr  = 0.2
)

// SIFT is the Scale-Invariant Feature Transform detector and descriptor. It
// builds a Gaussian scale-space pyramid, locates scale-space extrema of the
// Difference-of-Gaussians (DoG), refines them to sub-pixel/sub-scale position
// while rejecting low-contrast and edge responses, assigns one or more dominant
// gradient orientations, and computes the classic 128-dimensional gradient
// histogram descriptor. The resulting float descriptors are invariant to image
// translation, rotation and scale and are compared with the Euclidean distance
// ([NormL2]).
//
// The zero value is usable and applies the defaults; construct a customised
// instance with [NewSIFT].
//
// Differences from OpenCV: this implementation does not up-sample the input by
// 2× before building the pyramid (OpenCV's default doubles the image), so very
// fine keypoints near the original resolution limit are not detected and
// keypoint coordinates are reported directly in input pixels. Descriptors are
// returned as unit-length float vectors rather than the quantised uint8 form.
// The detected keypoint set is therefore close to, but not byte-identical with,
// OpenCV's.
type SIFT struct {
	// NFeatures caps the number of retained keypoints, keeping the strongest by
	// contrast response. Zero or negative keeps all of them.
	NFeatures int
	// NOctaveLayers is the number of scale layers sampled per octave. Zero means
	// the default (3).
	NOctaveLayers int
	// ContrastThreshold rejects weak DoG extrema. Zero means the default (0.04).
	ContrastThreshold float64
	// EdgeThreshold rejects edge-like extrema by the Hessian eigenvalue ratio.
	// Zero means the default (10).
	EdgeThreshold float64
	// Sigma is the blur of the first octave layer. Zero means the default (1.6).
	Sigma float64
}

// NewSIFT returns a SIFT detector retaining up to nFeatures keypoints (pass
// nFeatures <= 0 to keep all) with all other parameters at their defaults.
func NewSIFT(nFeatures int) *SIFT {
	return &SIFT{NFeatures: nFeatures}
}

func (s *SIFT) nOctaveLayers() int {
	if s.NOctaveLayers > 0 {
		return s.NOctaveLayers
	}
	return 3
}

func (s *SIFT) contrastThreshold() float64 {
	if s.ContrastThreshold > 0 {
		return s.ContrastThreshold
	}
	return 0.04
}

func (s *SIFT) edgeThreshold() float64 {
	if s.EdgeThreshold > 0 {
		return s.EdgeThreshold
	}
	return 10
}

func (s *SIFT) sigma() float64 {
	if s.Sigma > 0 {
		return s.Sigma
	}
	return 1.6
}

// Detect returns the SIFT keypoints of img without descriptors.
func (s *SIFT) Detect(img *cv.Mat) []KeyPoint {
	kps, _ := s.DetectAndCompute(img)
	return kps
}

// DetectAndCompute detects SIFT keypoints and computes their 128-dimensional
// float descriptors, returning parallel slices. The image may be single- or
// three-channel.
func (s *SIFT) DetectAndCompute(img *cv.Mat) ([]KeyPoint, [][]float64) {
	base := s.buildBaseImage(img)
	nOctaves := s.numOctaves(base)
	gpyr := s.buildGaussianPyramid(base, nOctaves)
	dpyr := s.buildDoGPyramid(gpyr)

	kps := s.findScaleSpaceExtrema(gpyr, dpyr)
	kps = dedupeAndRank(kps, s.NFeatures)

	desc := make([][]float64, len(kps))
	nLayers := s.nOctaveLayers()
	for i, kp := range kps {
		o, layer, scale := unpackOctave(kp)
		img := gpyr[o][layer]
		// Keypoint coordinates are in input pixels; bring them into the octave.
		xf := kp.Pt.X / (1 << uint(o))
		yf := kp.Pt.Y / (1 << uint(o))
		desc[i] = calcSIFTDescriptor(img, float64(xf), float64(yf), kp.Angle, scale, nLayers)
	}
	return kps, desc
}

// buildBaseImage converts img to a normalised (0..1) float image and applies the
// initial blur that brings its assumed input blur up to Sigma.
func (s *SIFT) buildBaseImage(img *cv.Mat) *fimage {
	f := fimageFromMat(img)
	for i := range f.data {
		f.data[i] /= 255.0
	}
	sigDiff := math.Sqrt(math.Max(s.sigma()*s.sigma()-siftInitSigma*siftInitSigma, 0.01))
	return f.gaussianBlur(sigDiff)
}

// numOctaves returns the number of octaves for an image, stopping when the
// image would become smaller than a few pixels.
func (s *SIFT) numOctaves(base *fimage) int {
	n := int(math.Round(math.Log2(float64(minInt(base.rows, base.cols)))-2)) + 1
	if n < 1 {
		n = 1
	}
	return n
}

// buildGaussianPyramid builds nOctaves octaves each of nOctaveLayers+3 Gaussian
// images with geometrically increasing blur.
func (s *SIFT) buildGaussianPyramid(base *fimage, nOctaves int) [][]*fimage {
	nLayers := s.nOctaveLayers()
	perOctave := nLayers + 3
	// Precompute the incremental sigmas within an octave.
	sig := make([]float64, perOctave)
	sig[0] = s.sigma()
	k := math.Pow(2, 1.0/float64(nLayers))
	for i := 1; i < perOctave; i++ {
		sigPrev := math.Pow(k, float64(i-1)) * s.sigma()
		sigTotal := sigPrev * k
		sig[i] = math.Sqrt(sigTotal*sigTotal - sigPrev*sigPrev)
	}

	pyr := make([][]*fimage, nOctaves)
	for o := 0; o < nOctaves; o++ {
		pyr[o] = make([]*fimage, perOctave)
		for i := 0; i < perOctave; i++ {
			switch {
			case o == 0 && i == 0:
				pyr[o][i] = base
			case i == 0:
				// Downsample the (nLayers)-th image of the previous octave.
				pyr[o][i] = pyr[o-1][nLayers].downsampleHalf()
			default:
				pyr[o][i] = pyr[o][i-1].gaussianBlur(sig[i])
			}
		}
	}
	return pyr
}

// buildDoGPyramid forms Difference-of-Gaussians images from adjacent Gaussian
// layers.
func (s *SIFT) buildDoGPyramid(gpyr [][]*fimage) [][]*fimage {
	dpyr := make([][]*fimage, len(gpyr))
	for o := range gpyr {
		n := len(gpyr[o]) - 1
		dpyr[o] = make([]*fimage, n)
		for i := 0; i < n; i++ {
			dpyr[o][i] = subtractF(gpyr[o][i+1], gpyr[o][i])
		}
	}
	return dpyr
}

// findScaleSpaceExtrema locates and refines DoG extrema across the pyramid,
// assigns orientations, and returns keypoints (with octave/layer/scale packed
// into Octave for the descriptor stage).
func (s *SIFT) findScaleSpaceExtrema(gpyr, dpyr [][]*fimage) []KeyPoint {
	nLayers := s.nOctaveLayers()
	prelim := math.Floor(0.5 * s.contrastThreshold() / float64(nLayers) * 255 * 0.00392156862745098) // /255
	var kps []KeyPoint
	for o := range dpyr {
		for layer := 1; layer <= nLayers; layer++ {
			cur := dpyr[o][layer]
			prev := dpyr[o][layer-1]
			next := dpyr[o][layer+1]
			rows, cols := cur.rows, cur.cols
			for y := siftImgBorder; y < rows-siftImgBorder; y++ {
				for x := siftImgBorder; x < cols-siftImgBorder; x++ {
					val := cur.data[y*cols+x]
					if math.Abs(val) <= prelim {
						continue
					}
					if !isExtremum(prev, cur, next, x, y, val) {
						continue
					}
					kp, li, ok := s.adjustLocalExtrema(dpyr, o, layer, x, y)
					if !ok {
						continue
					}
					// Assign orientations on the matching Gaussian image.
					scaleOctv := kp.Size * 0.5 / float64(uint(1)<<uint(o))
					gimg := gpyr[o][li]
					gx := int(math.Round(float64(kp.Pt.X) / float64(int(1)<<uint(o))))
					gy := int(math.Round(float64(kp.Pt.Y) / float64(int(1)<<uint(o))))
					hist, maxHist := calcOrientationHist(gimg, gx, gy,
						int(math.Round(siftOriRadius*scaleOctv)), siftOriSigFctr*scaleOctv, siftOriHistBins)
					magThr := maxHist * siftOriPeakRatio
					for j := 0; j < siftOriHistBins; j++ {
						l := (j - 1 + siftOriHistBins) % siftOriHistBins
						r := (j + 1) % siftOriHistBins
						if hist[j] > hist[l] && hist[j] > hist[r] && hist[j] >= magThr {
							// Parabolic peak interpolation.
							bin := float64(j) + 0.5*(hist[l]-hist[r])/(hist[l]-2*hist[j]+hist[r])
							bin = math.Mod(bin+siftOriHistBins, siftOriHistBins)
							angle := 360 - bin*360/siftOriHistBins
							if math.Abs(angle-360) < 1e-6 {
								angle = 0
							}
							okp := kp
							okp.Angle = angle
							kps = append(kps, okp)
						}
					}
				}
			}
		}
	}
	return kps
}

// isExtremum reports whether (x, y) in cur is strictly greater or strictly less
// than all 26 neighbours across the three DoG layers.
func isExtremum(prev, cur, next *fimage, x, y int, val float64) bool {
	cols := cur.cols
	if val > 0 {
		for dy := -1; dy <= 1; dy++ {
			for dx := -1; dx <= 1; dx++ {
				i := (y+dy)*cols + (x + dx)
				if cur.data[i] > val || prev.data[i] > val || next.data[i] > val {
					return false
				}
			}
		}
		return true
	}
	for dy := -1; dy <= 1; dy++ {
		for dx := -1; dx <= 1; dx++ {
			i := (y+dy)*cols + (x + dx)
			if cur.data[i] < val || prev.data[i] < val || next.data[i] < val {
				return false
			}
		}
	}
	return true
}

// adjustLocalExtrema refines an extremum to sub-pixel/sub-scale accuracy and
// applies the contrast and edge tests. On success it returns a keypoint (with
// input-pixel coordinates and Size set) and the refined layer index.
func (s *SIFT) adjustLocalExtrema(dpyr [][]*fimage, o, layer, x, y int) (KeyPoint, int, bool) {
	nLayers := s.nOctaveLayers()
	var xi, xr, xc float64
	i := 0
	for ; i < siftMaxInterp; i++ {
		cur := dpyr[o][layer]
		prev := dpyr[o][layer-1]
		next := dpyr[o][layer+1]
		cols := cur.cols

		at := func(im *fimage, yy, xx int) float64 { return im.data[yy*cols+xx] }
		dD := [3]float64{
			(at(cur, y, x+1) - at(cur, y, x-1)) * 0.5,
			(at(cur, y+1, x) - at(cur, y-1, x)) * 0.5,
			(at(next, y, x) - at(prev, y, x)) * 0.5,
		}
		v2 := 2 * at(cur, y, x)
		dxx := at(cur, y, x+1) + at(cur, y, x-1) - v2
		dyy := at(cur, y+1, x) + at(cur, y-1, x) - v2
		dss := at(next, y, x) + at(prev, y, x) - v2
		dxy := (at(cur, y+1, x+1) - at(cur, y+1, x-1) - at(cur, y-1, x+1) + at(cur, y-1, x-1)) * 0.25
		dxs := (at(next, y, x+1) - at(next, y, x-1) - at(prev, y, x+1) + at(prev, y, x-1)) * 0.25
		dys := (at(next, y+1, x) - at(next, y-1, x) - at(prev, y+1, x) + at(prev, y-1, x)) * 0.25

		H := [3][3]float64{
			{dxx, dxy, dxs},
			{dxy, dyy, dys},
			{dxs, dys, dss},
		}
		X, ok := solve3(H, [3]float64{-dD[0], -dD[1], -dD[2]})
		if !ok {
			return KeyPoint{}, 0, false
		}
		xc, xr, xi = X[0], X[1], X[2]
		if math.Abs(xi) < 0.5 && math.Abs(xr) < 0.5 && math.Abs(xc) < 0.5 {
			break
		}
		if math.Abs(xi) > float64(math.MaxInt32/3) || math.Abs(xr) > float64(math.MaxInt32/3) || math.Abs(xc) > float64(math.MaxInt32/3) {
			return KeyPoint{}, 0, false
		}
		x += int(math.Round(xc))
		y += int(math.Round(xr))
		layer += int(math.Round(xi))
		if layer < 1 || layer > nLayers ||
			x < siftImgBorder || x >= cur.cols-siftImgBorder ||
			y < siftImgBorder || y >= cur.rows-siftImgBorder {
			return KeyPoint{}, 0, false
		}
	}
	if i >= siftMaxInterp {
		return KeyPoint{}, 0, false
	}

	cur := dpyr[o][layer]
	prev := dpyr[o][layer-1]
	next := dpyr[o][layer+1]
	cols := cur.cols
	at := func(im *fimage, yy, xx int) float64 { return im.data[yy*cols+xx] }
	dD := [3]float64{
		(at(cur, y, x+1) - at(cur, y, x-1)) * 0.5,
		(at(cur, y+1, x) - at(cur, y-1, x)) * 0.5,
		(at(next, y, x) - at(prev, y, x)) * 0.5,
	}
	contr := at(cur, y, x) + 0.5*(dD[0]*xc+dD[1]*xr+dD[2]*xi)
	if math.Abs(contr)*float64(nLayers) < s.contrastThreshold() {
		return KeyPoint{}, 0, false
	}

	// Edge response test on the 2×2 spatial Hessian.
	v2 := 2 * at(cur, y, x)
	dxx := at(cur, y, x+1) + at(cur, y, x-1) - v2
	dyy := at(cur, y+1, x) + at(cur, y-1, x) - v2
	dxy := (at(cur, y+1, x+1) - at(cur, y+1, x-1) - at(cur, y-1, x+1) + at(cur, y-1, x-1)) * 0.25
	tr := dxx + dyy
	det := dxx*dyy - dxy*dxy
	et := s.edgeThreshold()
	if det <= 0 || tr*tr*et >= (et+1)*(et+1)*det {
		return KeyPoint{}, 0, false
	}

	scale := float64(int(1) << uint(o))
	kp := KeyPoint{
		Pt:       cv.Point{X: int(math.Round((float64(x) + xc) * scale)), Y: int(math.Round((float64(y) + xr) * scale))},
		Size:     s.sigma() * math.Pow(2, (float64(layer)+xi)/float64(nLayers)) * scale * 2,
		Response: math.Abs(contr),
		Octave:   packOctave(o, layer),
	}
	return kp, layer, true
}

// calcOrientationHist accumulates and smooths a gradient-orientation histogram
// in a circular window, returning the histogram and its peak value.
func calcOrientationHist(img *fimage, x, y, radius int, sigma float64, n int) ([]float64, float64) {
	if radius < 1 {
		radius = 1
	}
	expDenom := 2 * sigma * sigma
	temp := make([]float64, n)
	for dy := -radius; dy <= radius; dy++ {
		yy := y + dy
		if yy <= 0 || yy >= img.rows-1 {
			continue
		}
		for dx := -radius; dx <= radius; dx++ {
			xx := x + dx
			if xx <= 0 || xx >= img.cols-1 {
				continue
			}
			gx := img.at(yy, xx+1) - img.at(yy, xx-1)
			gy := img.at(yy-1, xx) - img.at(yy+1, xx)
			mag := math.Hypot(gx, gy)
			ang := math.Atan2(gy, gx) * 180 / math.Pi
			w := math.Exp(-float64(dx*dx+dy*dy) / expDenom)
			bin := int(math.Round(float64(n) * ang / 360))
			bin = (bin%n + n) % n
			temp[bin] += w * mag
		}
	}
	// Circular smoothing (OpenCV's 5-tap kernel).
	hist := make([]float64, n)
	get := func(i int) float64 { return temp[(i%n+n)%n] }
	maxVal := 0.0
	for i := 0; i < n; i++ {
		hist[i] = (get(i-2)+get(i+2))*(1.0/16) +
			(get(i-1)+get(i+1))*(4.0/16) +
			get(i)*(6.0/16)
		if hist[i] > maxVal {
			maxVal = hist[i]
		}
	}
	return hist, maxVal
}

// calcSIFTDescriptor computes the 128-d gradient-histogram descriptor of a
// keypoint at (x, y) in the given Gaussian image, oriented by angle (degrees) at
// the given octave scale, and returns a unit-length float vector.
func calcSIFTDescriptor(img *fimage, x, y, angle, scale float64, _ int) []float64 {
	d := siftDescrWidth
	n := siftDescrHistBin
	cosT := math.Cos(angle * math.Pi / 180)
	sinT := math.Sin(angle * math.Pi / 180)
	binsPerRad := float64(n) / 360
	expScale := -1.0 / (0.5 * float64(d) * float64(d))
	histWidth := siftDescrSclFctr * scale
	radius := int(math.Round(histWidth * math.Sqrt2 * float64(d+1) * 0.5))
	// Clamp the radius to the image diagonal.
	maxR := int(math.Sqrt(float64(img.rows*img.rows + img.cols*img.cols)))
	if radius > maxR {
		radius = maxR
	}
	cosT /= histWidth
	sinT /= histWidth

	hist := make([]float64, (d+2)*(d+2)*(n+2))
	for dy := -radius; dy <= radius; dy++ {
		for dx := -radius; dx <= radius; dx++ {
			// Rotate sample into the keypoint frame.
			cRot := float64(dx)*cosT + float64(dy)*sinT
			rRot := -float64(dx)*sinT + float64(dy)*cosT
			rbin := rRot + float64(d)/2 - 0.5
			cbin := cRot + float64(d)/2 - 0.5
			if rbin <= -1 || rbin >= float64(d) || cbin <= -1 || cbin >= float64(d) {
				continue
			}
			yy := int(math.Round(y)) + dy
			xx := int(math.Round(x)) + dx
			if yy <= 0 || yy >= img.rows-1 || xx <= 0 || xx >= img.cols-1 {
				continue
			}
			gx := img.at(yy, xx+1) - img.at(yy, xx-1)
			gy := img.at(yy-1, xx) - img.at(yy+1, xx)
			mag := math.Hypot(gx, gy)
			ang := math.Atan2(gy, gx) * 180 / math.Pi
			if ang < 0 {
				ang += 360
			}
			obin := (ang - angle) * binsPerRad
			w := math.Exp((cRot*cRot + rRot*rRot) * expScale)
			trilinear(hist, rbin, cbin, obin, mag*w, d, n)
		}
	}

	// Collapse the padded histogram into the dense d*d*n descriptor.
	dst := make([]float64, d*d*n)
	idx := 0
	for i := 0; i < d; i++ {
		for j := 0; j < d; j++ {
			for k := 0; k < n; k++ {
				dst[idx] = hist[((i+1)*(d+2)+(j+1))*(n+2)+k]
				idx++
			}
		}
	}
	normalizeDescriptor(dst)
	return dst
}

// trilinear distributes a weighted gradient sample into the 8 surrounding
// spatial/orientation histogram bins.
func trilinear(hist []float64, rbin, cbin, obin, mag float64, d, n int) {
	r0 := int(math.Floor(rbin))
	c0 := int(math.Floor(cbin))
	o0 := int(math.Floor(obin))
	rf := rbin - float64(r0)
	cf := cbin - float64(c0)
	of := obin - float64(o0)
	o0 = ((o0 % n) + n) % n

	for _, ri := range [2]int{0, 1} {
		rw := 1 - rf
		if ri == 1 {
			rw = rf
		}
		rr := r0 + ri
		if rr < -1 || rr >= d {
			continue
		}
		for _, ci := range [2]int{0, 1} {
			cw := 1 - cf
			if ci == 1 {
				cw = cf
			}
			cc := c0 + ci
			if cc < -1 || cc >= d {
				continue
			}
			for _, oi := range [2]int{0, 1} {
				ow := 1 - of
				if oi == 1 {
					ow = of
				}
				oo := (o0 + oi) % n
				bin := ((rr+1)*(d+2)+(cc+1))*(n+2) + oo
				hist[bin] += mag * rw * cw * ow
			}
		}
	}
}

// normalizeDescriptor applies SIFT's L2-normalise, clamp, renormalise sequence
// in place.
func normalizeDescriptor(v []float64) {
	norm := 0.0
	for _, x := range v {
		norm += x * x
	}
	norm = math.Sqrt(norm)
	if norm < 1e-12 {
		return
	}
	thr := norm * siftDescrMagThr
	norm2 := 0.0
	for i := range v {
		if v[i] > thr {
			v[i] = thr
		}
		norm2 += v[i] * v[i]
	}
	norm2 = math.Sqrt(norm2)
	if norm2 < 1e-12 {
		return
	}
	for i := range v {
		v[i] /= norm2
	}
}

// packOctave/unpackOctave/dedupeAndRank ---------------------------------------

// packOctave encodes the octave and layer into the KeyPoint.Octave field
// (octave in the low byte, layer in the next byte).
func packOctave(o, layer int) int {
	return o | (layer << 8)
}

// unpackOctave recovers the octave, layer and octave scale from a packed
// KeyPoint.
func unpackOctave(kp KeyPoint) (o, layer int, scale float64) {
	o = kp.Octave & 0xFF
	layer = (kp.Octave >> 8) & 0xFF
	scale = kp.Size * 0.5 / float64(int(1)<<uint(o))
	return
}

// dedupeAndRank removes duplicate keypoints, ranks them by descending response,
// and caps the count at nFeatures (when positive).
func dedupeAndRank(kps []KeyPoint, nFeatures int) []KeyPoint {
	sort.SliceStable(kps, func(i, j int) bool {
		if kps[i].Pt.X != kps[j].Pt.X {
			return kps[i].Pt.X < kps[j].Pt.X
		}
		if kps[i].Pt.Y != kps[j].Pt.Y {
			return kps[i].Pt.Y < kps[j].Pt.Y
		}
		if kps[i].Size != kps[j].Size {
			return kps[i].Size < kps[j].Size
		}
		if kps[i].Angle != kps[j].Angle {
			return kps[i].Angle < kps[j].Angle
		}
		return kps[i].Response > kps[j].Response
	})
	out := kps[:0]
	for i, kp := range kps {
		if i > 0 {
			p := kps[i-1]
			if p.Pt == kp.Pt && p.Size == kp.Size && p.Angle == kp.Angle {
				continue
			}
		}
		out = append(out, kp)
	}
	// Rank by response.
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Response != out[j].Response {
			return out[i].Response > out[j].Response
		}
		if out[i].Pt.Y != out[j].Pt.Y {
			return out[i].Pt.Y < out[j].Pt.Y
		}
		return out[i].Pt.X < out[j].Pt.X
	})
	if nFeatures > 0 && len(out) > nFeatures {
		out = out[:nFeatures]
	}
	return out
}

// solve3 solves the 3×3 linear system H x = b by Gaussian elimination with
// partial pivoting, returning ok=false when H is singular.
func solve3(H [3][3]float64, b [3]float64) ([3]float64, bool) {
	a := [3][4]float64{
		{H[0][0], H[0][1], H[0][2], b[0]},
		{H[1][0], H[1][1], H[1][2], b[1]},
		{H[2][0], H[2][1], H[2][2], b[2]},
	}
	for col := 0; col < 3; col++ {
		piv := col
		for r := col + 1; r < 3; r++ {
			if math.Abs(a[r][col]) > math.Abs(a[piv][col]) {
				piv = r
			}
		}
		if math.Abs(a[piv][col]) < 1e-18 {
			return [3]float64{}, false
		}
		a[col], a[piv] = a[piv], a[col]
		for r := 0; r < 3; r++ {
			if r == col {
				continue
			}
			f := a[r][col] / a[col][col]
			for c := col; c < 4; c++ {
				a[r][c] -= f * a[col][c]
			}
		}
	}
	return [3]float64{a[0][3] / a[0][0], a[1][3] / a[1][1], a[2][3] / a[2][2]}, true
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}
