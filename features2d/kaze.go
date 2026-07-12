package features2d

import (
	"math"
	"sort"

	cv "github.com/malcolmston/opencv"
)

// KAZE and AKAZE default parameters.
const (
	kazeDefaultOctaves   = 4
	kazeDefaultSublevels = 4
	kazeSigma0           = 1.6
	kazeDiffTau          = 0.25 // explicit-scheme time step (stable for [0,1] images)
	kazeDefaultThreshold = 0.001
)

// kazeLevel is one image of the nonlinear scale space, together with its
// gradients and the scale-normalised determinant-of-Hessian detector response.
type kazeLevel struct {
	octave   int     // octave index (image is downsampled by 2^octave)
	sublevel int     // sublevel within the octave
	sigmaOct float64 // Gaussian sigma in this octave's pixel grid
	esigma   float64 // equivalent sigma in the full-resolution grid
	L        *fimage // evolved image
	Lx, Ly   *fimage // first derivatives
	det      *fimage // scale-normalised |Hessian| response
}

// nonlinearScaleSpace holds the built evolution across octaves.
type nonlinearScaleSpace struct {
	levels    []kazeLevel
	sublevels int
}

// buildNonlinearScaleSpace constructs a KAZE/AKAZE nonlinear scale space. The
// input is normalised to [0,1]; a contrast factor is estimated once from the
// smoothed base image, and each octave is evolved with explicit nonlinear
// (Perona–Malik) diffusion, downsampling by two between octaves as in OpenCV's
// AKAZE. It returns the per-level images with their gradients and detector
// responses.
func buildNonlinearScaleSpace(img *cv.Mat, nOctaves, nSublevels int) *nonlinearScaleSpace {
	base := fimageFromMat(img)
	for i := range base.data {
		base.data[i] /= 255.0
	}
	base = base.gaussianBlur(kazeSigma0)
	k := computeContrastFactor(base)
	if k < 1e-6 {
		k = 1e-6
	}

	ss := &nonlinearScaleSpace{sublevels: nSublevels}
	octaveBase := base
	for o := 0; o < nOctaves; o++ {
		if octaveBase.rows < 8 || octaveBase.cols < 8 {
			break
		}
		prevT := 0.0
		cur := octaveBase.clone()
		for s := 0; s < nSublevels; s++ {
			sigmaOct := kazeSigma0 * math.Pow(2, float64(s)/float64(nSublevels))
			t := 0.5 * sigmaOct * sigmaOct
			if t > prevT {
				cur = evolveDiffusion(cur, k, t-prevT)
				prevT = t
			}
			lvl := kazeLevel{
				octave:   o,
				sublevel: s,
				sigmaOct: sigmaOct,
				esigma:   sigmaOct * float64(int(1)<<uint(o)),
				L:        cur.clone(),
			}
			lvl.Lx, lvl.Ly = gradients(lvl.L)
			lvl.det = determinantHessian(lvl.L, sigmaOct)
			ss.levels = append(ss.levels, lvl)
		}
		// Start the next octave from the last sublevel, downsampled.
		octaveBase = cur.downsampleHalf()
	}
	return ss
}

// computeContrastFactor estimates the Perona–Malik contrast parameter k as the
// 70th percentile of the gradient magnitude of the image (KAZE's method).
func computeContrastFactor(f *fimage) float64 {
	const nbins = 300
	hist := make([]int, nbins)
	maxGrad := 0.0
	mags := make([]float64, 0, f.rows*f.cols)
	for y := 1; y < f.rows-1; y++ {
		for x := 1; x < f.cols-1; x++ {
			gx, gy := f.gradXY(y, x)
			m := math.Hypot(gx, gy)
			mags = append(mags, m)
			if m > maxGrad {
				maxGrad = m
			}
		}
	}
	if maxGrad < 1e-9 {
		return 0.03
	}
	total := 0
	for _, m := range mags {
		bin := int(m / maxGrad * float64(nbins-1))
		hist[bin]++
		total++
	}
	threshold := int(0.7 * float64(total))
	acc, k := 0, nbins-1
	for i := 0; i < nbins; i++ {
		acc += hist[i]
		if acc >= threshold {
			k = i
			break
		}
	}
	return float64(k) / float64(nbins-1) * maxGrad
}

// evolveDiffusion advances the image by total diffusion time dt using explicit
// nonlinear (Perona–Malik) diffusion with conductivity recomputed each step.
func evolveDiffusion(L *fimage, k, dt float64) *fimage {
	steps := int(math.Ceil(dt / kazeDiffTau))
	if steps < 1 {
		steps = 1
	}
	tau := dt / float64(steps)
	k2 := k * k
	cur := L
	rows, cols := L.rows, L.cols
	for st := 0; st < steps; st++ {
		// Conductivity g2 = 1/(1 + |grad|^2/k^2).
		c := make([]float64, rows*cols)
		for y := 0; y < rows; y++ {
			for x := 0; x < cols; x++ {
				gx, gy := cur.gradXY(y, x)
				c[y*cols+x] = 1.0 / (1.0 + (gx*gx+gy*gy)/k2)
			}
		}
		out := newFImage(rows, cols)
		cAt := func(y, x int) float64 {
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
			return c[y*cols+x]
		}
		for y := 0; y < rows; y++ {
			for x := 0; x < cols; x++ {
				ci := c[y*cols+x]
				v := cur.at(y, x)
				fe := 0.5 * (ci + cAt(y, x+1)) * (cur.at(y, x+1) - v)
				fw := 0.5 * (ci + cAt(y, x-1)) * (cur.at(y, x-1) - v)
				fn := 0.5 * (ci + cAt(y-1, x)) * (cur.at(y-1, x) - v)
				fs := 0.5 * (ci + cAt(y+1, x)) * (cur.at(y+1, x) - v)
				out.data[y*cols+x] = v + tau*(fe+fw+fn+fs)
			}
		}
		cur = out
	}
	return cur
}

// gradients returns the central-difference first derivatives of f.
func gradients(f *fimage) (lx, ly *fimage) {
	lx = newFImage(f.rows, f.cols)
	ly = newFImage(f.rows, f.cols)
	for y := 0; y < f.rows; y++ {
		for x := 0; x < f.cols; x++ {
			gx, gy := f.gradXY(y, x)
			lx.data[y*f.cols+x] = gx
			ly.data[y*f.cols+x] = gy
		}
	}
	return
}

// determinantHessian returns the scale-normalised determinant of the Hessian,
// sigma^4 * (Lxx*Lyy - Lxy^2), the KAZE/AKAZE feature response.
func determinantHessian(f *fimage, sigma float64) *fimage {
	det := newFImage(f.rows, f.cols)
	norm := math.Pow(sigma, 4)
	for y := 0; y < f.rows; y++ {
		for x := 0; x < f.cols; x++ {
			v2 := 2 * f.at(y, x)
			lxx := f.at(y, x+1) + f.at(y, x-1) - v2
			lyy := f.at(y+1, x) + f.at(y-1, x) - v2
			lxy := (f.at(y+1, x+1) - f.at(y+1, x-1) - f.at(y-1, x+1) + f.at(y-1, x-1)) * 0.25
			det.data[y*f.cols+x] = norm * (lxx*lyy - lxy*lxy)
		}
	}
	return det
}

// detectKAZEKeypoints finds keypoints as scale-normalised Hessian extrema across
// the nonlinear scale space, assigns orientations, and returns them in
// descending response order (deduplicated). The level index each keypoint was
// found in is packed into KeyPoint.Octave for the descriptor stage.
func detectKAZEKeypoints(ss *nonlinearScaleSpace, threshold float64, nFeatures int) []KeyPoint {
	var kps []KeyPoint
	for li := range ss.levels {
		lvl := ss.levels[li]
		det := lvl.det
		rows, cols := det.rows, det.cols
		border := int(math.Ceil(lvl.sigmaOct)) + 1
		if border < 2 {
			border = 2
		}
		for y := border; y < rows-border; y++ {
			for x := border; x < cols-border; x++ {
				v := det.data[y*cols+x]
				if v <= threshold {
					continue
				}
				if !isSpatialMax(det, x, y, v) {
					continue
				}
				// Compare against the same location in adjacent sublevels of
				// this octave.
				if !isScaleMax(ss, li, x, y, v) {
					continue
				}
				scale := float64(int(1) << uint(lvl.octave))
				kp := KeyPoint{
					Pt:       cv.Point{X: int(math.Round(float64(x) * scale)), Y: int(math.Round(float64(y) * scale))},
					Size:     2 * lvl.esigma,
					Response: v,
					Octave:   li,
					Angle:    kazeOrientation(lvl, x, y),
				}
				kps = append(kps, kp)
			}
		}
	}
	kps = dedupeKAZE(kps)
	sort.SliceStable(kps, func(i, j int) bool {
		if kps[i].Response != kps[j].Response {
			return kps[i].Response > kps[j].Response
		}
		if kps[i].Pt.Y != kps[j].Pt.Y {
			return kps[i].Pt.Y < kps[j].Pt.Y
		}
		return kps[i].Pt.X < kps[j].Pt.X
	})
	if nFeatures > 0 && len(kps) > nFeatures {
		kps = kps[:nFeatures]
	}
	return kps
}

// isSpatialMax reports whether det[x,y] is the strict maximum of its 3×3
// neighbourhood.
func isSpatialMax(det *fimage, x, y int, v float64) bool {
	cols := det.cols
	for dy := -1; dy <= 1; dy++ {
		for dx := -1; dx <= 1; dx++ {
			if dx == 0 && dy == 0 {
				continue
			}
			if det.data[(y+dy)*cols+(x+dx)] >= v {
				return false
			}
		}
	}
	return true
}

// isScaleMax reports whether the response exceeds those at the same location in
// the neighbouring sublevels of the same octave.
func isScaleMax(ss *nonlinearScaleSpace, li, x, y int, v float64) bool {
	cur := ss.levels[li]
	for _, dl := range [2]int{-1, 1} {
		nl := li + dl
		if nl < 0 || nl >= len(ss.levels) {
			continue
		}
		if ss.levels[nl].octave != cur.octave {
			continue
		}
		det := ss.levels[nl].det
		if x < det.cols && y < det.rows && det.data[y*det.cols+x] >= v {
			return false
		}
	}
	return true
}

// dedupeKAZE removes keypoints at (almost) the same full-resolution location and
// scale that arise from overlapping octaves, keeping the strongest.
func dedupeKAZE(kps []KeyPoint) []KeyPoint {
	sort.SliceStable(kps, func(i, j int) bool { return kps[i].Response > kps[j].Response })
	var out []KeyPoint
	for _, kp := range kps {
		dup := false
		for _, o := range out {
			dx := float64(kp.Pt.X - o.Pt.X)
			dy := float64(kp.Pt.Y - o.Pt.Y)
			if math.Hypot(dx, dy) < 0.5*o.Size && math.Abs(kp.Size-o.Size) < 0.5*o.Size {
				dup = true
				break
			}
		}
		if !dup {
			out = append(out, kp)
		}
	}
	return out
}

// kazeOrientation estimates a dominant gradient orientation (degrees, [0,360))
// with the SURF sliding-window method on the level's gradients.
func kazeOrientation(lvl kazeLevel, x, y int) float64 {
	radius := int(math.Round(3 * lvl.sigmaOct))
	if radius < 1 {
		radius = 1
	}
	sigmaW := 2.5 * lvl.sigmaOct
	type sample struct{ ang, dx, dy float64 }
	var samples []sample
	for dy := -radius; dy <= radius; dy++ {
		for dx := -radius; dx <= radius; dx++ {
			if dx*dx+dy*dy > radius*radius {
				continue
			}
			yy, xx := y+dy, x+dx
			if yy < 0 || yy >= lvl.L.rows || xx < 0 || xx >= lvl.L.cols {
				continue
			}
			w := math.Exp(-float64(dx*dx+dy*dy) / (2 * sigmaW * sigmaW))
			gx := lvl.Lx.data[yy*lvl.L.cols+xx] * w
			gy := lvl.Ly.data[yy*lvl.L.cols+xx] * w
			samples = append(samples, sample{math.Atan2(gy, gx), gx, gy})
		}
	}
	best := 0.0
	bestAngle := 0.0
	const window = math.Pi / 3
	for a := 0.0; a < 2*math.Pi; a += 0.15 {
		var sx, sy float64
		for _, s := range samples {
			d := math.Mod(s.ang-a+2*math.Pi, 2*math.Pi)
			if d < window || d > 2*math.Pi-window {
				sx += s.dx
				sy += s.dy
			}
		}
		m := sx*sx + sy*sy
		if m > best {
			best = m
			bestAngle = math.Atan2(sy, sx)
		}
	}
	deg := bestAngle * 180 / math.Pi
	if deg < 0 {
		deg += 360
	}
	return deg
}

// KAZE is a multiscale 2D feature detector and descriptor operating in a
// nonlinear diffusion scale space. Unlike SIFT/SURF, which use Gaussian
// (linear) blurring that smooths across object boundaries, KAZE evolves the
// image with Perona–Malik nonlinear diffusion so that edges are preserved
// across scales, yielding more distinctive and better-localised features. It
// detects scale-normalised determinant-of-Hessian maxima and describes each
// keypoint with a rotation-invariant M-SURF-style 64-dimensional float
// descriptor (compared with [NormL2]).
//
// The zero value is usable and applies the defaults; construct a customised
// instance with [NewKAZE].
//
// Differences from OpenCV: the contrast factor uses the classic 70th-percentile
// gradient rule; diffusion uses a plain explicit scheme rather than OpenCV's
// Fast Explicit Diffusion (FED), so the evolved images — and hence the exact
// keypoint set — differ slightly, though the scale-space semantics match.
type KAZE struct {
	// NFeatures caps the retained keypoints (strongest first). Zero keeps all.
	NFeatures int
	// NOctaves is the number of octaves. Zero means the default (4).
	NOctaves int
	// NOctaveLayers is the number of sublevels per octave. Zero means the
	// default (4).
	NOctaveLayers int
	// Threshold is the detector response threshold. Zero means the default
	// (0.001).
	Threshold float64
}

// NewKAZE returns a KAZE detector retaining up to nFeatures keypoints (pass <= 0
// for all) with default scale-space parameters.
func NewKAZE(nFeatures int) *KAZE {
	return &KAZE{NFeatures: nFeatures}
}

func (a *KAZE) octaves() int {
	if a.NOctaves > 0 {
		return a.NOctaves
	}
	return kazeDefaultOctaves
}
func (a *KAZE) sublevels() int {
	if a.NOctaveLayers > 0 {
		return a.NOctaveLayers
	}
	return kazeDefaultSublevels
}
func (a *KAZE) threshold() float64 {
	if a.Threshold > 0 {
		return a.Threshold
	}
	return kazeDefaultThreshold
}

// Detect returns the KAZE keypoints of img without descriptors.
func (a *KAZE) Detect(img *cv.Mat) []KeyPoint {
	ss := buildNonlinearScaleSpace(img, a.octaves(), a.sublevels())
	return detectKAZEKeypoints(ss, a.threshold(), a.NFeatures)
}

// DetectAndCompute detects KAZE keypoints and computes their 64-dimensional
// float M-SURF descriptors, returning parallel slices.
func (a *KAZE) DetectAndCompute(img *cv.Mat) ([]KeyPoint, [][]float64) {
	ss := buildNonlinearScaleSpace(img, a.octaves(), a.sublevels())
	kps := detectKAZEKeypoints(ss, a.threshold(), a.NFeatures)
	desc := make([][]float64, len(kps))
	for i, kp := range kps {
		lvl := ss.levels[kp.Octave]
		scale := float64(int(1) << uint(lvl.octave))
		desc[i] = msurfDescriptor(lvl, float64(kp.Pt.X)/scale, float64(kp.Pt.Y)/scale, kp.Angle)
	}
	return kps, desc
}

// AKAZE is the Accelerated-KAZE detector and descriptor. It shares KAZE's
// nonlinear diffusion scale space and Hessian detector but produces a compact
// binary descriptor (a Modified-Local-Difference-Binary, M-LDB, string) that is
// matched with the Hamming distance ([NormHamming]) — much faster to store and
// compare than KAZE's float descriptor while remaining rotation- and
// scale-invariant.
//
// The zero value is usable and applies the defaults; construct a customised
// instance with [NewAKAZE]. The scale-space caveats noted for [KAZE] apply
// equally here.
type AKAZE struct {
	// NFeatures caps the retained keypoints (strongest first). Zero keeps all.
	NFeatures int
	// NOctaves is the number of octaves. Zero means the default (4).
	NOctaves int
	// NOctaveLayers is the number of sublevels per octave. Zero means the
	// default (4).
	NOctaveLayers int
	// Threshold is the detector response threshold. Zero means the default
	// (0.001).
	Threshold float64
}

// NewAKAZE returns an AKAZE detector retaining up to nFeatures keypoints (pass
// <= 0 for all) with default scale-space parameters.
func NewAKAZE(nFeatures int) *AKAZE {
	return &AKAZE{NFeatures: nFeatures}
}

func (a *AKAZE) octaves() int {
	if a.NOctaves > 0 {
		return a.NOctaves
	}
	return kazeDefaultOctaves
}
func (a *AKAZE) sublevels() int {
	if a.NOctaveLayers > 0 {
		return a.NOctaveLayers
	}
	return kazeDefaultSublevels
}
func (a *AKAZE) threshold() float64 {
	if a.Threshold > 0 {
		return a.Threshold
	}
	return kazeDefaultThreshold
}

// Detect returns the AKAZE keypoints of img without descriptors.
func (a *AKAZE) Detect(img *cv.Mat) []KeyPoint {
	ss := buildNonlinearScaleSpace(img, a.octaves(), a.sublevels())
	return detectKAZEKeypoints(ss, a.threshold(), a.NFeatures)
}

// DetectAndCompute detects AKAZE keypoints and computes their binary M-LDB
// descriptors, returning the keypoints and one bit-packed descriptor row each.
func (a *AKAZE) DetectAndCompute(img *cv.Mat) ([]KeyPoint, [][]byte) {
	ss := buildNonlinearScaleSpace(img, a.octaves(), a.sublevels())
	kps := detectKAZEKeypoints(ss, a.threshold(), a.NFeatures)
	desc := make([][]byte, len(kps))
	for i, kp := range kps {
		lvl := ss.levels[kp.Octave]
		scale := float64(int(1) << uint(lvl.octave))
		desc[i] = mldbDescriptor(lvl, float64(kp.Pt.X)/scale, float64(kp.Pt.Y)/scale, kp.Angle)
	}
	return kps, desc
}

// kazeSample gathers the rotated intensity and gradient at a patch location.
type kazeSample struct {
	u, v     float64 // normalised patch coordinates in [-1,1]
	val      float64 // intensity
	dxR, dyR float64 // gradient rotated into the keypoint frame
}

// samplePatch samples an N×N grid over a rotated square window of half-size W
// (in octave pixels) around (x, y), returning intensity and rotated gradients.
func samplePatch(lvl kazeLevel, x, y, angle, halfWin float64, n int) []kazeSample {
	cosT := math.Cos(angle * math.Pi / 180)
	sinT := math.Sin(angle * math.Pi / 180)
	cols := lvl.L.cols
	samples := make([]kazeSample, 0, n*n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			u := (float64(i)+0.5)/float64(n)*2 - 1
			v := (float64(j)+0.5)/float64(n)*2 - 1
			sx := u * halfWin
			sy := v * halfWin
			rx := sx*cosT - sy*sinT
			ry := sx*sinT + sy*cosT
			ix := int(math.Round(x + rx))
			iy := int(math.Round(y + ry))
			val := lvl.L.at(iy, ix)
			var gx, gy float64
			if iy >= 0 && iy < lvl.L.rows && ix >= 0 && ix < cols {
				gx = lvl.Lx.data[iy*cols+ix]
				gy = lvl.Ly.data[iy*cols+ix]
			}
			dxR := gx*cosT + gy*sinT
			dyR := -gx*sinT + gy*cosT
			samples = append(samples, kazeSample{u, v, val, dxR, dyR})
		}
	}
	return samples
}

// msurfDescriptor computes a 64-d rotation-invariant M-SURF-style float
// descriptor: a 4×4 grid of subregions, each summing (dx, dy, |dx|, |dy|) of the
// rotated gradient, Gaussian-weighted, then L2-normalised.
func msurfDescriptor(lvl kazeLevel, x, y, angle float64) []float64 {
	const grid = 4
	const nSamp = 20
	halfWin := 6 * lvl.sigmaOct
	if halfWin < 4 {
		halfWin = 4
	}
	samples := samplePatch(lvl, x, y, angle, halfWin, nSamp)
	desc := make([]float64, grid*grid*4)
	for _, s := range samples {
		cu := int((s.u + 1) / 2 * grid)
		cvIdx := int((s.v + 1) / 2 * grid)
		if cu < 0 {
			cu = 0
		} else if cu >= grid {
			cu = grid - 1
		}
		if cvIdx < 0 {
			cvIdx = 0
		} else if cvIdx >= grid {
			cvIdx = grid - 1
		}
		w := math.Exp(-(s.u*s.u + s.v*s.v) * 2)
		base := (cvIdx*grid + cu) * 4
		desc[base+0] += w * s.dxR
		desc[base+1] += w * s.dyR
		desc[base+2] += w * math.Abs(s.dxR)
		desc[base+3] += w * math.Abs(s.dyR)
	}
	// L2-normalise.
	var norm float64
	for _, v := range desc {
		norm += v * v
	}
	norm = math.Sqrt(norm)
	if norm > 1e-12 {
		for i := range desc {
			desc[i] /= norm
		}
	}
	return desc
}

// mldbGrids are the grid subdivisions of the M-LDB descriptor.
var mldbGrids = [3]int{2, 3, 4}

// mldbDescriptor computes a binary M-LDB descriptor: for each grid subdivision
// (2×2, 3×3, 4×4) it averages three channels (intensity, rotated dx, rotated dy)
// per cell and sets one bit for every ordered cell pair and channel where the
// first cell's average exceeds the second's. Bits are packed into bytes.
func mldbDescriptor(lvl kazeLevel, x, y, angle float64) []byte {
	const nSamp = 24
	halfWin := 6 * lvl.sigmaOct
	if halfWin < 4 {
		halfWin = 4
	}
	samples := samplePatch(lvl, x, y, angle, halfWin, nSamp)

	var bitsList []bool
	for _, g := range mldbGrids {
		cells := g * g
		sumV := make([]float64, cells)
		sumDx := make([]float64, cells)
		sumDy := make([]float64, cells)
		cnt := make([]int, cells)
		for _, s := range samples {
			cu := int((s.u + 1) / 2 * float64(g))
			cvv := int((s.v + 1) / 2 * float64(g))
			if cu < 0 {
				cu = 0
			} else if cu >= g {
				cu = g - 1
			}
			if cvv < 0 {
				cvv = 0
			} else if cvv >= g {
				cvv = g - 1
			}
			c := cvv*g + cu
			sumV[c] += s.val
			sumDx[c] += s.dxR
			sumDy[c] += s.dyR
			cnt[c]++
		}
		for c := 0; c < cells; c++ {
			if cnt[c] > 0 {
				n := float64(cnt[c])
				sumV[c] /= n
				sumDx[c] /= n
				sumDy[c] /= n
			}
		}
		for a := 0; a < cells; a++ {
			for b := a + 1; b < cells; b++ {
				bitsList = append(bitsList, sumV[a] > sumV[b])
				bitsList = append(bitsList, sumDx[a] > sumDx[b])
				bitsList = append(bitsList, sumDy[a] > sumDy[b])
			}
		}
	}
	out := make([]byte, (len(bitsList)+7)/8)
	for i, b := range bitsList {
		if b {
			out[i/8] |= 1 << uint(i%8)
		}
	}
	return out
}
