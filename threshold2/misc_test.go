package threshold2

import (
	"testing"

	cv "github.com/malcolmston/opencv"
)

func TestHistogram(t *testing.T) {
	img := twoBands(4, 4, 10, 20)
	h, err := ComputeHistogram(img)
	if err != nil {
		t.Fatal(err)
	}
	if h.Total != 16 {
		t.Fatalf("Total = %d, want 16", h.Total)
	}
	if h.Bins[10] != 8 || h.Bins[20] != 8 {
		t.Fatalf("bins[10]=%d bins[20]=%d, want 8/8", h.Bins[10], h.Bins[20])
	}
	if h.Mean() != 15 {
		t.Fatalf("Mean = %v, want 15", h.Mean())
	}
	first, last := h.Range()
	if first != 10 || last != 20 {
		t.Fatalf("Range = (%d,%d), want (10,20)", first, last)
	}
	if h.Peak() != 10 {
		t.Fatalf("Peak = %d, want 10 (lowest on tie)", h.Peak())
	}
	c := h.Cumulative()
	if c[9] != 0 || c[10] != 8 || c[255] != 16 {
		t.Fatalf("Cumulative unexpected: c[9]=%d c[10]=%d c[255]=%d", c[9], c[10], c[255])
	}
	d := h.Density()
	if d[10] != 0.5 || d[20] != 0.5 {
		t.Fatalf("Density = %v/%v, want 0.5/0.5", d[10], d[20])
	}
	if v := h.Variance(); v != 25 {
		t.Fatalf("Variance = %v, want 25", v)
	}
	if s := h.Smoothed(0); s.Bins != h.Bins {
		t.Fatal("Smoothed(0) should copy bins unchanged")
	}
}

func TestMultiOtsu(t *testing.T) {
	// Three equal bands at 30, 128 and 220.
	img := grayImage(6, 15, func(_, x int) uint8 {
		switch {
		case x < 5:
			return 30
		case x < 10:
			return 128
		default:
			return 220
		}
	})
	ts, err := MultiOtsu(img, 3)
	if err != nil {
		t.Fatal(err)
	}
	if len(ts) != 2 {
		t.Fatalf("got %d thresholds, want 2", len(ts))
	}
	if !(ts[0] >= 30 && ts[0] < 128 && ts[1] >= 128 && ts[1] < 220) {
		t.Fatalf("MultiOtsu thresholds = %v, want to separate 30/128/220", ts)
	}
	if ts[0] >= ts[1] {
		t.Fatalf("thresholds not ascending: %v", ts)
	}
	q, ts2, err := MultiOtsuQuantize(img, 3)
	if err != nil {
		t.Fatal(err)
	}
	if q.Rows != 6 || q.Cols != 15 || len(ts2) != 2 {
		t.Fatalf("quantize shape wrong: %dx%d ts=%v", q.Rows, q.Cols, ts2)
	}
	if _, err := MultiOtsu(img, 1); err == nil {
		t.Fatal("expected error for classes < 2")
	}
}

func TestMultiKapur(t *testing.T) {
	// Spread clusters so each class carries comparable entropy.
	img := grayImage(6, 15, func(y, x int) uint8 {
		switch {
		case x < 5:
			return uint8(28 + (y % 5))
		case x < 10:
			return uint8(126 + (y % 5))
		default:
			return uint8(218 + (y % 5))
		}
	})
	ts, err := MultiKapur(img, 3)
	if err != nil {
		t.Fatal(err)
	}
	if len(ts) != 2 || ts[0] >= ts[1] {
		t.Fatalf("MultiKapur = %v, want two ascending thresholds", ts)
	}
	for _, v := range ts {
		if v < 0 || v > 255 {
			t.Fatalf("MultiKapur threshold out of range: %v", ts)
		}
	}
	if _, _, err := MultiKapurQuantize(img, 3); err != nil {
		t.Fatal(err)
	}
	// A two-class MultiKapur should place its single threshold in the gap
	// between the two extreme clusters.
	two, err := MultiKapur(img, 2)
	if err != nil {
		t.Fatal(err)
	}
	if two[0] < 32 || two[0] > 218 {
		t.Fatalf("MultiKapur(2) = %v, want split between clusters", two)
	}
}

func TestOtsu2D(t *testing.T) {
	img := twoBands(8, 8, 40, 200)
	s, tt, err := Otsu2DThreshold(img)
	if err != nil {
		t.Fatal(err)
	}
	if s < 40 || s >= 200 {
		t.Fatalf("Otsu2D grey threshold = %d, want in [40,200)", s)
	}
	dst, _, _, err := Otsu2D(img, ObjectBright)
	if err != nil {
		t.Fatal(err)
	}
	// Interior corners are unambiguous.
	if dst.Data[0] != 0 {
		t.Fatalf("Otsu2D dark corner = %d, want 0", dst.Data[0])
	}
	if dst.Data[7] != 255 {
		t.Fatalf("Otsu2D bright corner = %d, want 255", dst.Data[7])
	}
	_ = tt
}

func TestHysteresis(t *testing.T) {
	// One row: strong seed connects a chain of weak pixels; the last pixel is
	// below the low threshold and stays background.
	img := grayImage(1, 4, func(_, x int) uint8 {
		return []uint8{255, 100, 100, 30}[x]
	})
	dst, err := Hysteresis(img, 50, 200)
	if err != nil {
		t.Fatal(err)
	}
	want := []uint8{255, 255, 255, 0}
	for x := 0; x < 4; x++ {
		if dst.Data[x] != want[x] {
			t.Fatalf("Hysteresis[%d] = %d, want %d", x, dst.Data[x], want[x])
		}
	}
	// Isolated weak pixels with no strong seed must all be background.
	weak := grayImage(1, 3, func(_, _ int) uint8 { return 100 })
	dst2, _ := Hysteresis(weak, 50, 200)
	for x := 0; x < 3; x++ {
		if dst2.Data[x] != 0 {
			t.Fatalf("isolated weak[%d] = %d, want 0", x, dst2.Data[x])
		}
	}
	if _, err := Hysteresis(img, 200, 50); err == nil {
		t.Fatal("expected error when low > high")
	}
}

func TestPerChannelAndInRange(t *testing.T) {
	// 2x2 RGB image with distinct channel values.
	img := cv.NewMat(2, 2, 3)
	set := func(p int, r, g, b uint8) {
		img.Data[p*3+0] = r
		img.Data[p*3+1] = g
		img.Data[p*3+2] = b
	}
	set(0, 10, 200, 10)
	set(1, 250, 10, 10)
	set(2, 10, 10, 250)
	set(3, 250, 250, 250)

	dst, thr, err := PerChannelOtsu(img, ObjectBright)
	if err != nil {
		t.Fatal(err)
	}
	if len(thr) != 3 || dst.Channels != 3 {
		t.Fatalf("PerChannelOtsu shape wrong: thr=%v ch=%d", thr, dst.Channels)
	}

	mask, err := InRange(img, []uint8{200, 200, 200}, []uint8{255, 255, 255})
	if err != nil {
		t.Fatal(err)
	}
	// Only the all-250 pixel is inside the range.
	want := []uint8{0, 0, 0, 255}
	for i := 0; i < 4; i++ {
		if mask.Data[i] != want[i] {
			t.Fatalf("InRange[%d] = %d, want %d", i, mask.Data[i], want[i])
		}
	}
	if _, err := InRange(img, []uint8{0}, []uint8{255}); err == nil {
		t.Fatal("expected error for wrong bound length")
	}
	if _, err := PerChannelThreshold(img, []int{1, 2}, ObjectBright); err == nil {
		t.Fatal("expected error for wrong threshold length")
	}
}

func TestToGrayLuma(t *testing.T) {
	img := cv.NewMat(1, 1, 3)
	img.Data[0] = 255 // R
	img.Data[1] = 0   // G
	img.Data[2] = 0   // B
	g, err := ToGray(img)
	if err != nil {
		t.Fatal(err)
	}
	// Rec.601 red weight ~0.299 -> 76 or 77.
	if g.Data[0] < 74 || g.Data[0] > 78 {
		t.Fatalf("ToGray red luma = %d, want ~76", g.Data[0])
	}
}
