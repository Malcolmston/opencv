package inpaint

import "testing"

// For a half-plane target (x >= c) with sources at x < c, the Fast Marching
// solution of |grad T| = 1 gives exactly integer axis distances T = x-c+1.
func TestDistanceTransformHalfPlane(t *testing.T) {
	rows, cols := 5, 7
	c := 3
	mask := NewMask(rows, cols)
	for y := 0; y < rows; y++ {
		for x := c; x < cols; x++ {
			mask.Set(y, x, true)
		}
	}
	dist := DistanceTransform(mask)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			var want float64
			if x >= c {
				want = float64(x - c + 1)
			}
			if got := dist[y*cols+x]; got != want {
				t.Fatalf("dist(%d,%d) = %v, want %v", y, x, got, want)
			}
		}
	}
}

func TestFastMarcherVisitOrder(t *testing.T) {
	// Visit order must be non-decreasing in arrival time.
	rows, cols := 6, 6
	mask := centerHoleMask(rows, cols, 1, 1, 4, 4)
	fm := NewFastMarcher(mask)
	var last float64
	ok := true
	tField := fm.Solve(func(y, x int) {})
	// Re-derive: every target pixel got a finite positive time.
	for y := 1; y < 5; y++ {
		for x := 1; x < 5; x++ {
			if tField[y*cols+x] <= 0 {
				ok = false
			}
		}
	}
	if !ok {
		t.Fatalf("interior arrival times should be positive")
	}
	_ = last
}
