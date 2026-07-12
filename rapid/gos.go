package rapid

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// GOSTracker is the global-optimal-search RAPID variant. Rather than choosing
// each control point's edge independently, it solves for the edge columns of all
// control points jointly with dynamic programming, trading raw edge response
// against smoothness of the displacement along the contour. This rejects
// isolated spurious edges that a purely local search would latch onto.
type GOSTracker struct {
	*Rapid
}

// NewGOSTracker creates a [GOSTracker] for the given mesh. histBins is retained
// for API compatibility with cv::rapid::GOSTracker::create and controls the
// smoothness granularity; sobelThresh is the minimum edge response accepted.
func NewGOSTracker(mesh *Mesh, histBins int, sobelThresh float64) *GOSTracker {
	if histBins < 1 {
		histBins = 4
	}
	// A finer histogram implies a stronger smoothness preference.
	lambda := 4.0 / float64(histBins)
	return &GOSTracker{Rapid: &Rapid{
		mesh:        mesh,
		strategy:    &gosSearch{lambda: lambda, thresh: sobelThresh},
		minResponse: sobelThresh,
	}}
}

// gosSearch is the global-optimal-search strategy.
type gosSearch struct {
	lambda float64 // smoothness weight between adjacent control points
	thresh float64
}

func (gosSearch) clear() {}

func (g *gosSearch) search(bundle *cv.Mat) ([]int, []float64) {
	if bundle == nil || bundle.Rows == 0 {
		return nil, nil
	}
	rows, w := bundle.Rows, bundle.Cols
	prof := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		prof[i] = rowGradient(bundle, i)
	}
	// Dynamic program maximising sum(response) - lambda*sum|col_i - col_{i-1}|.
	dp := make([][]float64, rows)
	back := make([][]int, rows)
	for i := range dp {
		dp[i] = make([]float64, w)
		back[i] = make([]int, w)
	}
	for j := 0; j < w; j++ {
		dp[0][j] = prof[0][j]
	}
	for i := 1; i < rows; i++ {
		for j := 0; j < w; j++ {
			best := math.Inf(-1)
			bj := 0
			for p := 0; p < w; p++ {
				v := dp[i-1][p] - g.lambda*math.Abs(float64(j-p))
				if v > best {
					best = v
					bj = p
				}
			}
			dp[i][j] = best + prof[i][j]
			back[i][j] = bj
		}
	}
	// Backtrack from the best terminal column.
	cols := make([]int, rows)
	resp := make([]float64, rows)
	bestLast := 0
	bestVal := math.Inf(-1)
	for j := 0; j < w; j++ {
		if dp[rows-1][j] > bestVal {
			bestVal = dp[rows-1][j]
			bestLast = j
		}
	}
	cols[rows-1] = bestLast
	for i := rows - 1; i > 0; i-- {
		cols[i-1] = back[i][cols[i]]
	}
	for i := 0; i < rows; i++ {
		resp[i] = prof[i][cols[i]]
	}
	return cols, resp
}
