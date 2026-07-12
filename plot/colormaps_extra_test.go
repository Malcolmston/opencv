package plot

import "testing"

func TestExtraColormapEndpoints(t *testing.T) {
	cases := []struct {
		name   string
		cm     Colormap
		lo, hi [3]uint8
	}{
		{"autumn", ColormapAutumn, [3]uint8{255, 0, 0}, [3]uint8{255, 255, 0}},
		{"winter", ColormapWinter, [3]uint8{0, 0, 255}, [3]uint8{0, 255, 128}},
		{"summer", ColormapSummer, [3]uint8{0, 128, 102}, [3]uint8{255, 255, 102}},
		{"spring", ColormapSpring, [3]uint8{255, 0, 255}, [3]uint8{255, 255, 0}},
		{"ocean", ColormapOcean, [3]uint8{0, 32, 64}, [3]uint8{255, 255, 255}},
		{"rainbow", ColormapRainbow, [3]uint8{255, 0, 0}, [3]uint8{255, 0, 255}},
		{"pink", ColormapPink, [3]uint8{30, 15, 15}, [3]uint8{255, 255, 255}},
		{"parula", ColormapParula, [3]uint8{53, 42, 135}, [3]uint8{249, 251, 21}},
		{"magma", ColormapMagma, [3]uint8{0, 0, 4}, [3]uint8{252, 253, 191}},
		{"inferno", ColormapInferno, [3]uint8{0, 0, 4}, [3]uint8{252, 255, 164}},
		{"cividis", ColormapCividis, [3]uint8{0, 32, 76}, [3]uint8{255, 233, 69}},
		{"twilight", ColormapTwilight, [3]uint8{226, 217, 226}, [3]uint8{226, 217, 226}},
		{"turbo", ColormapTurbo, [3]uint8{48, 18, 59}, [3]uint8{122, 4, 3}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			table := Table(tc.cm)
			if len(table) != 256 {
				t.Fatalf("table length = %d, want 256", len(table))
			}
			if table[0] != tc.lo {
				t.Errorf("entry 0 = %v, want %v", table[0], tc.lo)
			}
			if table[255] != tc.hi {
				t.Errorf("entry 255 = %v, want %v", table[255], tc.hi)
			}
			// The map must not be a constant colour.
			constant := true
			for i := 1; i < 256; i++ {
				if table[i] != table[0] {
					constant = false
					break
				}
			}
			if constant {
				t.Errorf("%s colormap is constant", tc.name)
			}
		})
	}
}

func TestTableDelegatesToOriginal(t *testing.T) {
	// Table must reproduce the original eight maps exactly.
	orig := ColormapTable(ColormapJet)
	got := Table(ColormapJet)
	for i := range orig {
		if got[i] != orig[i] {
			t.Fatalf("Table(Jet)[%d] = %v, want %v", i, got[i], orig[i])
		}
	}
}

func TestColorizeAppliesExtraMap(t *testing.T) {
	src := rampMat()
	out := Colorize(src, ColormapTurbo)
	if out.Channels != 3 || out.Cols != 256 {
		t.Fatalf("shape = %dx%dx%d", out.Rows, out.Cols, out.Channels)
	}
	table := Table(ColormapTurbo)
	if got := px3(out, 0, 0); got != table[0] {
		t.Errorf("intensity 0 = %v, want %v", got, table[0])
	}
	if got := px3(out, 0, 255); got != table[255] {
		t.Errorf("intensity 255 = %v, want %v", got, table[255])
	}
}

func TestTablePanicsOnUnknown(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic on unknown colormap")
		}
	}()
	Table(Colormap(9999))
}
