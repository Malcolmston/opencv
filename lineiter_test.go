package cv

import "testing"

func TestLineIteratorHorizontal(t *testing.T) {
	it := NewLineIterator(0, 0, 2, 0)
	if it.Count() != 3 {
		t.Fatalf("count = %d, want 3", it.Count())
	}
	pts := it.Points()
	want := []Point{{0, 0}, {1, 0}, {2, 0}}
	if len(pts) != 3 {
		t.Fatalf("pts = %v", pts)
	}
	for i := range want {
		if pts[i] != want[i] {
			t.Errorf("pts[%d] = %v, want %v", i, pts[i], want[i])
		}
	}
}

func TestLineIteratorDiagonal(t *testing.T) {
	it := NewLineIterator(0, 0, 2, 2)
	pts := it.Points()
	want := []Point{{0, 0}, {1, 1}, {2, 2}}
	if len(pts) != len(want) {
		t.Fatalf("len = %d, want %d (%v)", len(pts), len(want), pts)
	}
	for i := range want {
		if pts[i] != want[i] {
			t.Errorf("pts[%d] = %v, want %v", i, pts[i], want[i])
		}
	}
}

func TestLineIteratorManualWalk(t *testing.T) {
	it := NewLineIterator(3, 3, 0, 0)
	n := 0
	for it.Valid() {
		n++
		it.Next()
	}
	if n != 4 {
		t.Errorf("walked %d pixels, want 4", n)
	}
}
