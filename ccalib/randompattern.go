package ccalib

import (
	"math/rand"

	cv "github.com/malcolmston/opencv"
)

// RandomPatternGenerator synthesises a random-dot calibration target: a set of
// bright filled discs scattered on a dark background. Such targets are matched
// by the distinctive local geometry of the dots rather than by a regular grid,
// so they can be detected even when only partially visible. Generation is fully
// deterministic given the seed.
type RandomPatternGenerator struct {
	width, height int
	seed          int64
	numDots       int
	centers       [][2]float64
}

// NewRandomPatternGenerator returns a generator producing a width×height,
// single-channel pattern of numDots dots, seeded with seed for reproducibility.
// It panics on non-positive dimensions or dot count.
func NewRandomPatternGenerator(width, height, numDots int, seed int64) *RandomPatternGenerator {
	if width <= 0 || height <= 0 || numDots <= 0 {
		panic("ccalib: NewRandomPatternGenerator requires positive dimensions and dot count")
	}
	return &RandomPatternGenerator{width: width, height: height, numDots: numDots, seed: seed}
}

// Generate renders the pattern and returns it as a single-channel [cv.Mat] with
// bright dots (value 255) on a black background. Repeated calls with the same
// generator produce identical output. The dot centres are recorded and exposed
// through [RandomPatternGenerator.Centers].
func (g *RandomPatternGenerator) Generate() *cv.Mat {
	rng := rand.New(rand.NewSource(g.seed))
	img := cv.NewMat(g.height, g.width, 1)
	g.centers = g.centers[:0]
	margin := 12
	minR, maxR := 4, 9
	placed := 0
	attempts := 0
	minSep := float64(maxR*2 + 8)
	for placed < g.numDots && attempts < g.numDots*200 {
		attempts++
		cx := margin + rng.Intn(g.width-2*margin)
		cy := margin + rng.Intn(g.height-2*margin)
		// Reject dots that would overlap an existing one so components stay
		// separable by connected-component labelling.
		ok := true
		for _, c := range g.centers {
			dx := c[0] - float64(cx)
			dy := c[1] - float64(cy)
			if dx*dx+dy*dy < minSep*minSep {
				ok = false
				break
			}
		}
		if !ok {
			continue
		}
		r := minR + rng.Intn(maxR-minR+1)
		cv.Circle(img, cv.Point{X: cx, Y: cy}, r, cv.NewScalar(255), -1)
		g.centers = append(g.centers, [2]float64{float64(cx), float64(cy)})
		placed++
	}
	return img
}

// Centers returns the dot centres of the most recently generated pattern, in
// pixel coordinates. The slice is nil before the first [RandomPatternGenerator.Generate] call.
func (g *RandomPatternGenerator) Centers() [][2]float64 {
	out := make([][2]float64, len(g.centers))
	copy(out, g.centers)
	return out
}

// RandomPatternCornerFinder matches captured views of a random-dot target
// against a reference image of that target, producing the 3D object points and
// their observed 2D image points needed to calibrate a camera. The physical
// width and height describe the printed target so that object points come out in
// real-world units (e.g. millimetres).
type RandomPatternCornerFinder struct {
	physWidth  float64
	physHeight float64
	minMatches int
	tol        float64

	refBlobs []blob
	refDesc  []descriptor
	refW     int
	refH     int
}

// NewRandomPatternCornerFinder creates a finder for a target whose printed size
// is physWidth×physHeight (in any consistent unit). minMatches is the minimum
// number of consistent dot correspondences required to accept a view; pass 0 for
// the default of 10.
func NewRandomPatternCornerFinder(physWidth, physHeight float64, minMatches int) *RandomPatternCornerFinder {
	if minMatches <= 0 {
		minMatches = 10
	}
	return &RandomPatternCornerFinder{
		physWidth:  physWidth,
		physHeight: physHeight,
		minMatches: minMatches,
		tol:        6.0,
	}
}

// LoadPattern registers the reference image of the target, detecting its dots
// and building their descriptors. It reports whether enough dots were found to
// support matching.
func (f *RandomPatternCornerFinder) LoadPattern(pattern *cv.Mat) bool {
	f.refW = pattern.Cols
	f.refH = pattern.Rows
	f.refBlobs = detectBlobs(pattern, 6, 0)
	f.refDesc = buildDescriptors(f.refBlobs)
	return len(f.refBlobs) >= f.minMatches
}

// LoadPatternBlobs registers the reference dots directly from known centres
// (for instance those returned by [RandomPatternGenerator.Centers]), skipping
// detection. refWidth and refHeight are the reference image dimensions used to
// scale object points. It reports whether enough dots were supplied.
func (f *RandomPatternCornerFinder) LoadPatternBlobs(centers [][2]float64, refWidth, refHeight int) bool {
	f.refW = refWidth
	f.refH = refHeight
	f.refBlobs = make([]blob, len(centers))
	for i, c := range centers {
		f.refBlobs[i] = blob{X: c[0], Y: c[1], R: 5}
	}
	f.refDesc = buildDescriptors(f.refBlobs)
	return len(f.refBlobs) >= f.minMatches
}

// ComputeObjectImagePoints detects the target in the observation image and
// returns the matched object points (3D, with Z = 0, in the physical units of
// the printed target) and their observed image points (pixels). ok is false when
// the target is not confidently found. LoadPattern (or LoadPatternBlobs) must be
// called first.
func (f *RandomPatternCornerFinder) ComputeObjectImagePoints(observation *cv.Mat) (objPts [][3]float64, imgPts [][2]float64, ok bool) {
	if len(f.refBlobs) == 0 {
		return nil, nil, false
	}
	queryBlobs := detectBlobs(observation, 6, 0)
	queryDesc := buildDescriptors(queryBlobs)
	matches := matchBlobs(f.refDesc, queryDesc)
	inliers, _, okH := filterMatchesByHomography(f.refBlobs, queryBlobs, matches, f.tol)
	if !okH || len(inliers) < f.minMatches {
		return nil, nil, false
	}
	sx := f.physWidth / float64(f.refW)
	sy := f.physHeight / float64(f.refH)
	for _, m := range inliers {
		rb := f.refBlobs[m.ref]
		qb := queryBlobs[m.query]
		objPts = append(objPts, [3]float64{rb.X * sx, rb.Y * sy, 0})
		imgPts = append(imgPts, [2]float64{qb.X, qb.Y})
	}
	return objPts, imgPts, true
}
