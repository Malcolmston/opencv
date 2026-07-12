package rapid

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// searchStrategy locates the edge column and response for every row of a line
// bundle. Implementations may keep state between calls (cleared by
// [Tracker.ClearState]).
type searchStrategy interface {
	search(bundle *cv.Mat) (cols []int, response []float64)
	clear()
}

// Tracker is the common interface implemented by all RAPID trackers: [Rapid],
// [OLSTracker] and [GOSTracker].
type Tracker interface {
	// Compute iterates the tracker from an initial pose until term is met,
	// returning the refined pose and the ratio of control points that yielded a
	// correspondence on the final iteration.
	Compute(img *cv.Mat, num, length int, k [3][3]float64, pose Pose, term TermCriteria) (Pose, float64)
	// Proceed performs a single RAPID iteration and returns the updated pose,
	// the ratio of control points matched, and the RMS reprojection error
	// (perpendicular residual) measured before the update.
	Proceed(img *cv.Mat, num, length int, k [3][3]float64, pose Pose) (Pose, float64, float64)
	// ClearState resets any per-run state accumulated by the tracker.
	ClearState()
}

// Rapid is the basic RAPID tracker. It searches each contour normal for the
// strongest intensity edge and refines the pose with a single Gauss-Newton
// step per iteration.
type Rapid struct {
	mesh        *Mesh
	strategy    searchStrategy
	minResponse float64
}

// NewRapid creates a [Rapid] tracker for the given mesh. It mirrors
// cv::rapid::Rapid::create.
func NewRapid(mesh *Mesh) *Rapid {
	return &Rapid{mesh: mesh, strategy: &basicSearch{}, minResponse: 10}
}

// basicSearch selects the strongest edge per bundle row and holds no state.
type basicSearch struct{}

func (basicSearch) search(bundle *cv.Mat) ([]int, []float64) { return FindCorrespondencies(bundle) }
func (basicSearch) clear()                                   {}

// ClearState resets the tracker's search state.
func (r *Rapid) ClearState() {
	if r.strategy != nil {
		r.strategy.clear()
	}
}

// Proceed performs a single RAPID iteration: extract control points, read the
// line bundles, find the strongest edges, build correspondences, and apply one
// Gauss-Newton pose update. It returns the updated pose, the ratio of control
// points matched, and the RMS perpendicular residual measured at the input pose.
func (r *Rapid) Proceed(img *cv.Mat, num, length int, k [3][3]float64, pose Pose) (Pose, float64, float64) {
	ctl := ExtractControlPoints(num, length, r.mesh, pose, k, img.Cols, img.Rows)
	if len(ctl) == 0 {
		return pose, 0, 0
	}
	bundle, locs := ExtractLineBundle(length, ctl, img)
	cols, resp := r.strategy.search(bundle)
	matches, _ := ConvertCorrespondencies(cols, resp, locs, ctl, r.minResponse)
	ratio := float64(len(matches)) / float64(len(ctl))
	if len(matches) < 3 {
		return pose, ratio, 0
	}
	newPose, rmsd, ok := gaussNewtonUpdate(pose, matches, k)
	if !ok {
		return pose, ratio, rmsd
	}
	return newPose, ratio, rmsd
}

// Compute iterates [Rapid.Proceed] until the pose increment falls below
// term.Epsilon or term.MaxCount iterations have run. It returns the refined pose
// and the correspondence ratio of the final iteration.
func (r *Rapid) Compute(img *cv.Mat, num, length int, k [3][3]float64, pose Pose, term TermCriteria) (Pose, float64) {
	iters := term.MaxCount
	if iters < 1 {
		iters = 1
	}
	ratio := 0.0
	for it := 0; it < iters; it++ {
		np, rr, _ := r.Proceed(img, num, length, k, pose)
		ratio = rr
		d := poseDelta(np, pose)
		pose = np
		if d < term.Epsilon {
			break
		}
	}
	return pose, ratio
}

// poseDelta returns the combined magnitude of the rotation and translation
// difference between two poses.
func poseDelta(a, b Pose) float64 {
	var s float64
	for i := 0; i < 3; i++ {
		dr := a.Rvec[i] - b.Rvec[i]
		dt := a.Tvec[i] - b.Tvec[i]
		s += dr*dr + dt*dt
	}
	return math.Sqrt(s)
}

// gaussNewtonUpdate solves one Gauss-Newton step for the 6-DOF pose from a set
// of one-dimensional (along-normal) correspondences. The residual for each match
// is the signed perpendicular distance between the projected model point and the
// located image edge; the Jacobian projects the 2×6 pinhole projection Jacobian
// onto the search normal. It returns the updated pose, the RMS residual at the
// input pose, and whether the linear system was solvable.
func gaussNewtonUpdate(pose Pose, matches []Correspondence, k [3][3]float64) (Pose, float64, bool) {
	r := rodrigues(pose.Rvec)
	kk := intrinsics(k)
	a := make([][]float64, 6)
	for i := range a {
		a[i] = make([]float64, 6)
	}
	b := make([]float64, 6)
	var sse float64
	var cnt int
	for _, m := range matches {
		proj, xc, ok := project(m.Object, r, pose.Tvec, kk)
		if !ok {
			continue
		}
		rx := [3]float64{xc[0] - pose.Tvec[0], xc[1] - pose.Tvec[1], xc[2] - pose.Tvec[2]}
		j := projectionJacobian(xc, rx, kk)
		var jr [6]float64
		for c := 0; c < 6; c++ {
			jr[c] = m.Normal.X*j[0][c] + m.Normal.Y*j[1][c]
		}
		e := m.Normal.dot(proj.sub(m.Image))
		sse += e * e
		cnt++
		for aa := 0; aa < 6; aa++ {
			for cc := 0; cc < 6; cc++ {
				a[aa][cc] += jr[aa] * jr[cc]
			}
			b[aa] += -e * jr[aa]
		}
	}
	if cnt < 3 {
		return pose, 0, false
	}
	// Levenberg-style damping for numerical stability.
	for i := 0; i < 6; i++ {
		a[i][i] += 1e-6 * (a[i][i] + 1)
	}
	rmsd := math.Sqrt(sse / float64(cnt))
	delta, ok := solveSPD(a, b, 6)
	if !ok {
		return pose, rmsd, false
	}
	dOmega := [3]float64{delta[0], delta[1], delta[2]}
	rNew := mul3(rodrigues(dOmega), r)
	newPose := Pose{
		Rvec: rotationToRvec(rNew),
		Tvec: [3]float64{pose.Tvec[0] + delta[3], pose.Tvec[1] + delta[4], pose.Tvec[2] + delta[5]},
	}
	return newPose, rmsd, true
}

// rowGradient returns the central-difference magnitude profile of bundle row i.
func rowGradient(bundle *cv.Mat, i int) []float64 {
	w := bundle.Cols
	g := make([]float64, w)
	for j := 0; j < w; j++ {
		switch {
		case j == 0:
			g[j] = math.Abs(float64(bundle.At(i, 1, 0)) - float64(bundle.At(i, 0, 0)))
		case j == w-1:
			g[j] = math.Abs(float64(bundle.At(i, w-1, 0)) - float64(bundle.At(i, w-2, 0)))
		default:
			g[j] = math.Abs(float64(bundle.At(i, j+1, 0)) - float64(bundle.At(i, j-1, 0)))
		}
	}
	return g
}
