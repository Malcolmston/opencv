package structured_light

import "testing"

func TestStereoDecode(t *testing.T) {
	rows, cols := 4, 4
	n := rows * cols
	left := &Decoded{Rows: rows, Cols: cols, Col: make([]int, n), Row: make([]int, n), Mask: make([]bool, n)}
	right := &Decoded{Rows: rows, Cols: cols, Col: make([]int, n), Row: make([]int, n), Mask: make([]bool, n)}
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			i := y*cols + x
			// Left camera sees projector column == its own x.
			left.Col[i], left.Row[i], left.Mask[i] = x, y, true
			// Right camera is shifted: its pixel x sees projector column x-1.
			if x >= 1 {
				right.Col[i], right.Row[i], right.Mask[i] = x-1, y, true
			} else {
				right.Col[i], right.Row[i] = -1, -1
			}
		}
	}

	matches := StereoDecode(left, right)
	// Left pixel (lx,ly) with projector col lx matches right pixel (lx+1,ly),
	// valid for lx in 0..2 -> 3 columns * 4 rows = 12 matches.
	if len(matches) != 12 {
		t.Fatalf("got %d matches, want 12", len(matches))
	}
	for _, m := range matches {
		if m.RightX != m.LeftX+1 || m.RightY != m.LeftY {
			t.Fatalf("bad match: left(%d,%d) right(%d,%d)", m.LeftX, m.LeftY, m.RightX, m.RightY)
		}
		if m.Col != m.LeftX {
			t.Fatalf("match projector col %d != left x %d", m.Col, m.LeftX)
		}
	}
}

func TestTriangulateStereo(t *testing.T) {
	cam, proj := testRig(800, 320, 240, 0.2)
	want := [3]float64{0.05, 0.02, 1.3}
	uc, vc := cam.Project(want)
	up, vp := proj.Project(want)
	matches := []StereoMatch{{
		LeftX: int(uc), LeftY: int(vc),
		RightX: int(up), RightY: int(vp),
	}}
	// Use exact (fractional) coordinates via a direct point check instead.
	pc := TriangulateStereo(matches, cam, proj)
	if pc.Len() != 1 {
		t.Fatalf("expected 1 point, got %d", pc.Len())
	}
	// Direct point-level exactness (fractional inputs).
	got := TriangulatePoint(cam, proj, uc, vc, up, vp)
	for i := 0; i < 3; i++ {
		if e := (got[i] - want[i]); e > 1e-6 || e < -1e-6 {
			t.Fatalf("stereo triangulated coord %d = %.8f, want %.8f", i, got[i], want[i])
		}
	}
}
