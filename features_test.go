package cv

import (
	"math"
	"testing"
)

// nearAny reports whether p is within dist of any point in pts.
func nearAny(p Point, pts []Point, dist float64) bool {
	for _, q := range pts {
		if math.Hypot(float64(p.X-q.X), float64(p.Y-q.Y)) <= dist {
			return true
		}
	}
	return false
}

func TestCornerHarrisPeaksAtCorner(t *testing.T) {
	m := synthSquare(40, 10, 10, 20)
	resp := CornerHarris(m, 3, 3, 0.04)
	_, _, _, _, maxX, maxY := MinMaxLoc(resp)
	corners := []Point{{10, 10}, {29, 10}, {29, 29}, {10, 29}}
	if !nearAny(Point{maxX, maxY}, corners, 3) {
		t.Errorf("Harris max at (%d,%d) not near a square corner", maxX, maxY)
	}
}

func TestGoodFeaturesToTrackFindsCorners(t *testing.T) {
	m := synthSquare(60, 15, 15, 30)
	pts := GoodFeaturesToTrack(m, 10, 0.1, 5, 3)
	if len(pts) < 4 {
		t.Fatalf("found %d corners, want >= 4", len(pts))
	}
	corners := []Point{{15, 15}, {44, 15}, {44, 44}, {15, 44}}
	found := 0
	for _, c := range corners {
		if nearAny(c, pts, 4) {
			found++
		}
	}
	if found < 4 {
		t.Errorf("only %d/4 square corners detected, points=%v", found, pts)
	}
}

func TestHoughLinesVerticalLine(t *testing.T) {
	m := NewMat(40, 40, 1)
	for y := 0; y < 40; y++ {
		m.Set(y, 15, 0, 255)
	}
	lines := HoughLines(m, 1, math.Pi/180, 30)
	if len(lines) == 0 {
		t.Fatal("no lines detected")
	}
	// A vertical line x=15 has theta≈0 and rho≈15.
	top := lines[0]
	theta := top.Theta
	if theta > math.Pi/2 {
		theta = math.Pi - theta // fold near-180 onto near-0
	}
	if theta > 3*math.Pi/180 {
		t.Errorf("line theta = %v rad, want ~0", top.Theta)
	}
	if math.Abs(math.Abs(top.Rho)-15) > 1.5 {
		t.Errorf("line rho = %v, want ~15", top.Rho)
	}
}

func TestHoughLinesPHorizontalSegment(t *testing.T) {
	m := NewMat(40, 40, 1)
	for x := 5; x <= 30; x++ {
		m.Set(20, x, 0, 255)
	}
	segs := HoughLinesP(m, 1, math.Pi/180, 15, 20, 3)
	if len(segs) == 0 {
		t.Fatal("no segments detected")
	}
	best := segs[0]
	length := math.Hypot(float64(best.Pt2.X-best.Pt1.X), float64(best.Pt2.Y-best.Pt1.Y))
	if length < 20 {
		t.Errorf("segment length = %v, want >= 20", length)
	}
	// The segment lies on row 20.
	if best.Pt1.Y != 20 || best.Pt2.Y != 20 {
		t.Errorf("segment not on row 20: %+v", best)
	}
}

func TestHoughCirclesDrawnCircle(t *testing.T) {
	m := NewMat(60, 60, 1)
	Circle(m, Point{30, 30}, 12, NewScalar(255), 1)
	circles := HoughCircles(m, 20, 100, 15, 8, 16)
	if len(circles) == 0 {
		t.Fatal("no circles detected")
	}
	c := circles[0]
	if math.Hypot(float64(c.X-30), float64(c.Y-30)) > 4 {
		t.Errorf("circle center (%d,%d) not near (30,30)", c.X, c.Y)
	}
	if c.Radius < 9 || c.Radius > 15 {
		t.Errorf("circle radius = %d, want ~12", c.Radius)
	}
}

func TestFASTCornersSquare(t *testing.T) {
	m := synthSquare(40, 12, 12, 16)
	corners := FASTCorners(m, 20, true)
	if len(corners) == 0 {
		t.Fatal("FAST found no corners")
	}
	sqCorners := []Point{{12, 12}, {27, 12}, {27, 27}, {12, 27}}
	found := 0
	for _, c := range sqCorners {
		if nearAny(c, corners, 3) {
			found++
		}
	}
	if found < 4 {
		t.Errorf("FAST detected %d/4 corners, points=%v", found, corners)
	}
}
