package features2d

import "testing"

func TestHammingDistance(t *testing.T) {
	cases := []struct {
		a, b []byte
		want int
	}{
		{[]byte{0x00}, []byte{0x00}, 0},
		{[]byte{0xFF}, []byte{0x00}, 8},
		{[]byte{0b1010}, []byte{0b0101}, 4},
		{[]byte{0xFF, 0x0F}, []byte{0xF0, 0xFF}, 8},
	}
	for _, c := range cases {
		if got := HammingDistance(c.a, c.b); got != c.want {
			t.Errorf("HammingDistance(%v,%v)=%d want %d", c.a, c.b, got, c.want)
		}
	}
}

func TestHammingDistanceLengthMismatchPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic on length mismatch")
		}
	}()
	HammingDistance([]byte{0}, []byte{0, 0})
}

func TestBFMatcherHammingExact(t *testing.T) {
	// Query 0 is identical to train 1; query 1 is identical to train 0.
	q := NewBinaryDescriptors([][]byte{{0xAA}, {0x0F}})
	tr := NewBinaryDescriptors([][]byte{{0x0F}, {0xAA}})
	m := NewBFMatcher(NormHamming)
	matches := m.Match(q, tr)
	if len(matches) != 2 {
		t.Fatalf("expected 2 matches, got %d", len(matches))
	}
	if matches[0].TrainIdx != 1 || matches[0].Distance != 0 {
		t.Errorf("query 0 -> %+v, want TrainIdx 1 distance 0", matches[0])
	}
	if matches[1].TrainIdx != 0 || matches[1].Distance != 0 {
		t.Errorf("query 1 -> %+v, want TrainIdx 0 distance 0", matches[1])
	}
}

func TestBFMatcherL2(t *testing.T) {
	q := NewFloatDescriptors([][]float64{{0, 0}})
	tr := NewFloatDescriptors([][]float64{{3, 4}, {1, 0}})
	m := NewBFMatcher(NormL2)
	matches := m.Match(q, tr)
	if len(matches) != 1 {
		t.Fatalf("expected 1 match, got %d", len(matches))
	}
	// Nearest is train 1 at distance 1 (vs train 0 at distance 5).
	if matches[0].TrainIdx != 1 || matches[0].Distance != 1 {
		t.Errorf("got %+v, want TrainIdx 1 distance 1", matches[0])
	}
}

func TestBFMatcherCrossCheck(t *testing.T) {
	// Train 0 is every query's nearest, but only query 0 is train 0's nearest,
	// so cross-check must keep just that mutual pair.
	q := NewBinaryDescriptors([][]byte{{0x00}, {0x03}})
	tr := NewBinaryDescriptors([][]byte{{0x00}, {0xFF}})
	m := &BFMatcher{Norm: NormHamming, CrossCheck: true}
	matches := m.Match(q, tr)
	if len(matches) != 1 {
		t.Fatalf("cross-check expected 1 mutual match, got %d", len(matches))
	}
	if matches[0].QueryIdx != 0 || matches[0].TrainIdx != 0 {
		t.Errorf("got %+v, want query 0 -> train 0", matches[0])
	}
}

func TestKnnMatchOrdering(t *testing.T) {
	q := NewBinaryDescriptors([][]byte{{0x00}})
	tr := NewBinaryDescriptors([][]byte{{0xFF}, {0x01}, {0x0F}})
	m := NewBFMatcher(NormHamming)
	knn := m.KnnMatch(q, tr, 2)
	if len(knn) != 1 || len(knn[0]) != 2 {
		t.Fatalf("expected 1 row of 2, got %d rows", len(knn))
	}
	// Distances: train0=8, train1=1, train2=4 -> nearest train1 then train2.
	if knn[0][0].TrainIdx != 1 || knn[0][0].Distance != 1 {
		t.Errorf("first knn = %+v, want train 1 distance 1", knn[0][0])
	}
	if knn[0][1].TrainIdx != 2 || knn[0][1].Distance != 4 {
		t.Errorf("second knn = %+v, want train 2 distance 4", knn[0][1])
	}
}

func TestRatioTestUnit(t *testing.T) {
	knn := [][]DMatch{
		{{Distance: 1}, {Distance: 10}}, // clear: 1 < 0.75*10 -> keep
		{{Distance: 8}, {Distance: 10}}, // ambiguous: 8 >= 7.5 -> drop
		{{Distance: 3}},                 // single candidate -> keep
		{},                              // empty -> skip
	}
	got := RatioTest(knn, 0.75)
	if len(got) != 2 {
		t.Fatalf("expected 2 survivors, got %d: %+v", len(got), got)
	}
	if got[0].Distance != 1 || got[1].Distance != 3 {
		t.Errorf("unexpected survivors: %+v", got)
	}
}

func TestBFMatcherWrongTypePanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic passing float descriptors to a Hamming matcher")
		}
	}()
	m := NewBFMatcher(NormHamming)
	m.Match(NewFloatDescriptors([][]float64{{1}}), NewFloatDescriptors([][]float64{{1}}))
}
