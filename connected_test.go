package cv

import (
	"math"
	"testing"
)

func TestConnectedComponentsTwoBlobs(t *testing.T) {
	m := NewMat(20, 30, 1)
	for y := 2; y < 7; y++ {
		for x := 2; x < 7; x++ {
			m.Set(y, x, 0, 255)
		}
	}
	for y := 10; y < 15; y++ {
		for x := 20; x < 25; x++ {
			m.Set(y, x, 0, 255)
		}
	}
	labels, count := ConnectedComponents(m, Connectivity8)
	// count includes the background label 0.
	if count != 3 {
		t.Fatalf("component count = %d, want 3 (bg + 2)", count)
	}
	l1 := labels[4*30+4]
	l2 := labels[12*30+22]
	if l1 == 0 || l2 == 0 || l1 == l2 {
		t.Errorf("blob labels l1=%d l2=%d, want distinct non-zero", l1, l2)
	}
}

func TestConnectedComponents4vs8(t *testing.T) {
	// Two pixels touching only diagonally: one component under 8, two under 4.
	m := NewMat(5, 5, 1)
	m.Set(1, 1, 0, 255)
	m.Set(2, 2, 0, 255)
	_, c8 := ConnectedComponents(m, Connectivity8)
	if c8 != 2 {
		t.Errorf("8-connectivity count = %d, want 2 (bg+1)", c8)
	}
	_, c4 := ConnectedComponents(m, Connectivity4)
	if c4 != 3 {
		t.Errorf("4-connectivity count = %d, want 3 (bg+2)", c4)
	}
}

func TestConnectedComponentsWithStats(t *testing.T) {
	m := NewMat(20, 20, 1)
	for y := 4; y < 9; y++ { // 5x5 block at (4,4)..(8,8)
		for x := 4; x < 9; x++ {
			m.Set(y, x, 0, 255)
		}
	}
	_, count, stats := ConnectedComponentsWithStats(m, Connectivity8)
	if count != 2 {
		t.Fatalf("count = %d, want 2", count)
	}
	s := stats[1]
	if s.Area != 25 {
		t.Errorf("area = %d, want 25", s.Area)
	}
	if s.Rect.X != 4 || s.Rect.Y != 4 || s.Rect.Width != 5 || s.Rect.Height != 5 {
		t.Errorf("bbox = %+v, want {4 4 5 5}", s.Rect)
	}
	if math.Abs(s.CentroidX-6) > 1e-9 || math.Abs(s.CentroidY-6) > 1e-9 {
		t.Errorf("centroid = (%v,%v), want (6,6)", s.CentroidX, s.CentroidY)
	}
	// Background stats cover the rest.
	if stats[0].Area != 400-25 {
		t.Errorf("background area = %d, want %d", stats[0].Area, 375)
	}
}
