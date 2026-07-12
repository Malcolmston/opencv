package cudafeatures2d

import (
	"testing"
)

func TestMatchHammingExact(t *testing.T) {
	q := DescriptorsToGpuMat([][]byte{{0xAA}, {0x0F}})
	tr := DescriptorsToGpuMat([][]byte{{0x0F}, {0xAA}})
	m := CreateBFMatcher(NormHamming)
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

func TestKnnMatch(t *testing.T) {
	q := DescriptorsToGpuMat([][]byte{{0x00}})
	tr := DescriptorsToGpuMat([][]byte{{0xFF}, {0x01}, {0x0F}})
	m := CreateBFMatcher(NormHamming)
	knn := m.KnnMatch(q, tr, 2)
	if len(knn) != 1 || len(knn[0]) != 2 {
		t.Fatalf("expected 1x2 knn result, got %v", knn)
	}
	// Nearest to 0x00 is 0x01 (1 bit), then 0x0F (4 bits).
	if knn[0][0].TrainIdx != 1 || knn[0][0].Distance != 1 {
		t.Errorf("nearest = %+v, want train 1 distance 1", knn[0][0])
	}
	if knn[0][1].TrainIdx != 2 || knn[0][1].Distance != 4 {
		t.Errorf("second = %+v, want train 2 distance 4", knn[0][1])
	}
}

func TestRadiusMatch(t *testing.T) {
	q := DescriptorsToGpuMat([][]byte{{0x00}})
	tr := DescriptorsToGpuMat([][]byte{{0xFF}, {0x01}, {0x0F}})
	m := CreateBFMatcher(NormHamming)
	// Radius 4 keeps 0x01 (1) and 0x0F (4), excludes 0xFF (8).
	rad := m.RadiusMatch(q, tr, 4)
	if len(rad) != 1 {
		t.Fatalf("expected 1 query row, got %d", len(rad))
	}
	if len(rad[0]) != 2 {
		t.Fatalf("expected 2 matches within radius, got %d: %v", len(rad[0]), rad[0])
	}
	for _, dm := range rad[0] {
		if dm.Distance > 4 {
			t.Errorf("radius match beyond limit: %+v", dm)
		}
	}
}

func TestMatchConvert(t *testing.T) {
	q := DescriptorsToGpuMat([][]byte{{0x00}, {0xFF}})
	tr := DescriptorsToGpuMat([][]byte{{0x01}, {0xFE}})
	m := CreateBFMatcher(NormHamming)
	knn := m.KnnMatch(q, tr, 2)
	flat := m.MatchConvert(knn)
	if len(flat) != 2 {
		t.Fatalf("MatchConvert produced %d matches, want 2", len(flat))
	}
	// Best of query 0 is train 0 (0x00 vs 0x01 = 1 bit).
	if flat[0].QueryIdx != 0 || flat[0].TrainIdx != 0 {
		t.Errorf("flat[0] = %+v, want query 0 -> train 0", flat[0])
	}
	// Empty rows are skipped.
	if len(m.MatchConvert([][]DMatch{{}, {}})) != 0 {
		t.Error("MatchConvert should skip empty rows")
	}
}

func TestMatcherL2Path(t *testing.T) {
	// With NormL2 each descriptor byte is treated as a float sample.
	q := DescriptorsToGpuMat([][]byte{{0, 0}})
	tr := DescriptorsToGpuMat([][]byte{{3, 4}, {1, 0}})
	m := CreateBFMatcher(NormL2)
	matches := m.Match(q, tr)
	if len(matches) != 1 {
		t.Fatalf("expected 1 match, got %d", len(matches))
	}
	// Nearest is train 1 at distance 1 (vs train 0 at distance 5).
	if matches[0].TrainIdx != 1 || matches[0].Distance != 1 {
		t.Errorf("got %+v, want TrainIdx 1 distance 1", matches[0])
	}
}

func TestMatcherEmptyInputs(t *testing.T) {
	m := CreateBFMatcher(NormHamming)
	empty := &GpuMat{}
	full := DescriptorsToGpuMat([][]byte{{0x00}})
	if m.Match(empty, full) != nil {
		t.Error("Match with empty query should be nil")
	}
	if m.Match(full, empty) != nil {
		t.Error("Match with empty train should be nil")
	}
	if m.KnnMatch(empty, full, 1) != nil {
		t.Error("KnnMatch with empty query should be nil")
	}
	if m.RadiusMatch(empty, full, 1) != nil {
		t.Error("RadiusMatch with empty query should be nil")
	}
}

func TestDescriptorsGpuMatEmpty(t *testing.T) {
	if !DescriptorsToGpuMat(nil).Empty() {
		t.Error("nil rows should give empty GpuMat")
	}
	if !DescriptorsToGpuMat([][]byte{}).Empty() {
		t.Error("no rows should give empty GpuMat")
	}
	if !DescriptorsToGpuMat([][]byte{{}}).Empty() {
		t.Error("zero-width rows should give empty GpuMat")
	}
}

func TestDescriptorsToGpuMatRaggedPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic on ragged rows")
		}
	}()
	DescriptorsToGpuMat([][]byte{{0x00, 0x01}, {0x02}})
}

func TestEmptyImagePanics(t *testing.T) {
	cases := []struct {
		name string
		fn   func()
	}{
		{"orb-detect", func() { CreateORB(10).Detect(&GpuMat{}, nil) }},
		{"orb-compute", func() { CreateORB(10).Compute(&GpuMat{}, []KeyPoint{{}}) }},
		{"fast", func() { CreateFastFeatureDetector(20, true).Detect(&GpuMat{}, nil) }},
		{"corners", func() { CreateGoodFeaturesToTrackDetector(10, 0.01, 3, 3).Detect(&GpuMat{}, nil) }},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatalf("%s: expected panic on empty image", c.name)
				}
			}()
			c.fn()
		})
	}
}

func TestORBComputeEmptyKeypoints(t *testing.T) {
	img := NewGpuMat(checkerboard(48, 12))
	_, desc := CreateORB(10).Compute(img, nil)
	if !desc.Empty() {
		t.Error("Compute with no keypoints should give empty descriptors")
	}
}
