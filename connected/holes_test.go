package connected

import "testing"

func TestFillHolesRing(t *testing.T) {
	img := matFromRows([]string{
		".....",
		".###.",
		".#.#.",
		".###.",
		".....",
	})
	// One enclosed hole at (2,2).
	if h := CountHoles(img, Conn8); h != 1 {
		t.Errorf("hole count = %d, want 1", h)
	}
	filled := FillHoles(img, Conn8)
	// The 3x3 ring becomes a solid 3x3 block: 9 foreground pixels.
	if countValue(filled, 255) != 9 {
		t.Errorf("filled fg = %d, want 9", countValue(filled, 255))
	}
	if filled.Data[2*5+2] != 255 {
		t.Errorf("hole not filled")
	}
}

func TestHolesMask(t *testing.T) {
	img := matFromRows([]string{
		".....",
		".###.",
		".#.#.",
		".###.",
		".....",
	})
	hm := HolesMask(img, Conn8)
	if countValue(hm, 255) != 1 {
		t.Errorf("holes mask fg = %d, want 1", countValue(hm, 255))
	}
	if hm.Data[2*5+2] != 255 {
		t.Errorf("hole pixel not marked")
	}
}

func TestEulerNumber(t *testing.T) {
	// Two separate solid blobs, no holes: Euler number 2.
	solid := matFromRows([]string{
		"#..#",
		"#..#",
	})
	if e := EulerNumber(solid, Conn4); e != 2 {
		t.Errorf("Euler (two blobs) = %d, want 2", e)
	}
	// One ring: 1 component, 1 hole -> Euler 0.
	ring := matFromRows([]string{
		"###",
		"#.#",
		"###",
	})
	if e := EulerNumber(ring, Conn8); e != 0 {
		t.Errorf("Euler (ring) = %d, want 0", e)
	}
}

func TestNoHolesWhenOpenToborder(t *testing.T) {
	// A 'C' shape: the interior connects to the border, so no hole.
	c := matFromRows([]string{
		"###",
		"#..",
		"###",
	})
	if h := CountHoles(c, Conn8); h != 0 {
		t.Errorf("C-shape holes = %d, want 0", h)
	}
}
