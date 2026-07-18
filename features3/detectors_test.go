package features3

import (
	"testing"

	cv "github.com/malcolmston/opencv"
)

func TestFASTDetectsSquareCorner(t *testing.T) {
	img := filledSquare(41, 41, 10, 10, 30, 30, 255)
	kps := FASTKeyPoints(img, 20, true)
	if len(kps) == 0 {
		t.Fatal("FAST found no corners on a square")
	}
	corners := [][2]float64{{10, 10}, {30, 10}, {10, 30}, {30, 30}}
	found := 0
	for _, c := range corners {
		if nearAnyKeyPoint(kps, c[0], c[1], 2) {
			found++
		}
	}
	if found == 0 {
		t.Fatal("FAST found no keypoint near any square corner")
	}
}

func TestIsFASTCornerAtCorner(t *testing.T) {
	img := filledSquare(41, 41, 10, 10, 30, 30, 255)
	// The top-left corner of the bright square must pass the segment test.
	if !IsFASTCorner(img, 10, 10, 20) {
		t.Fatal("expected FAST corner at (10,10)")
	}
	// A point deep inside the uniform square is not a corner.
	if IsFASTCorner(img, 20, 20, 20) {
		t.Fatal("did not expect FAST corner in a flat region")
	}
}

func TestFASTScoreBorderZero(t *testing.T) {
	img := filledSquare(41, 41, 10, 10, 30, 30, 255)
	if FASTScore(img, 0, 0) != 0 {
		t.Fatal("border FAST score should be 0")
	}
	if FASTScore(img, 10, 10) <= 0 {
		t.Fatal("corner FAST score should be positive")
	}
}

func TestAGASTDetectsCorner(t *testing.T) {
	img := filledSquare(41, 41, 10, 10, 30, 30, 255)
	kps := AGASTKeyPoints(img, 20, true)
	if len(kps) == 0 {
		t.Fatal("AGAST found no corners")
	}
	if !nearAnyKeyPoint(kps, 10, 10, 2) && !nearAnyKeyPoint(kps, 30, 30, 2) {
		t.Fatal("AGAST found no keypoint near a corner")
	}
}

func TestAGASTScoreAtLeastThreshold(t *testing.T) {
	img := filledSquare(41, 41, 10, 10, 30, 30, 255)
	s := AGASTScore(img, 10, 10, 20)
	if s < 20 {
		t.Fatalf("AGAST adaptive score %d should be >= base threshold 20", s)
	}
	if IsAGASTCorner(img, 20, 20, 20) {
		t.Fatal("did not expect AGAST corner in flat region")
	}
}

func TestSUSANCornersOnSquare(t *testing.T) {
	img := filledSquare(41, 41, 12, 12, 28, 28, 255)
	kps := SUSANCorners(img, 3, 40, 0, 0.3, 0)
	if len(kps) == 0 {
		t.Fatal("SUSAN found no corners")
	}
	corners := [][2]float64{{12, 12}, {28, 12}, {12, 28}, {28, 28}}
	found := 0
	for _, c := range corners {
		if nearAnyKeyPoint(kps, c[0], c[1], 3) {
			found++
		}
	}
	if found == 0 {
		t.Fatal("SUSAN found no keypoint near any corner")
	}
}

func TestKeypointNMS(t *testing.T) {
	kps := []KeyPoint{
		NewKeyPoint(0, 0, 5),
		NewKeyPoint(1, 0, 3),
		NewKeyPoint(10, 10, 4),
	}
	kept := KeypointNMS(kps, 3)
	if len(kept) != 2 {
		t.Fatalf("KeypointNMS kept %d, want 2", len(kept))
	}
	// Strongest first, and the weak neighbour of (0,0) is gone.
	if kept[0].Response != 5 || kept[1].Response != 4 {
		t.Fatalf("unexpected survivors: %+v", kept)
	}
}

func TestNonMaxSuppressionFloat(t *testing.T) {
	resp := cv.NewFloatMat(5, 5)
	resp.Data[1*5+1] = 5
	resp.Data[3*5+3] = 9
	resp.Data[3*5+2] = 4 // dominated neighbour of the (3,3) peak
	kps := NonMaxSuppressionFloat(resp, 1, 1)
	if len(kps) != 2 {
		t.Fatalf("expected 2 maxima, got %d: %+v", len(kps), kps)
	}
	if kps[0].Pt.X != 3 || kps[0].Pt.Y != 3 {
		t.Fatalf("strongest maximum wrong: %+v", kps[0])
	}
	pts := LocalMaxima(resp, 1, 1)
	if len(pts) != 2 {
		t.Fatalf("LocalMaxima returned %d", len(pts))
	}
}

func TestKeypointUtilities(t *testing.T) {
	kps := []KeyPoint{
		NewKeyPoint(1, 1, 2),
		NewKeyPoint(2, 2, 9),
		NewKeyPoint(3, 3, 5),
	}
	best := RetainBest(kps, 2)
	if len(best) != 2 || best[0].Response != 9 || best[1].Response != 5 {
		t.Fatalf("RetainBest wrong: %+v", best)
	}
	SortKeyPointsByResponse(kps)
	if kps[0].Response != 9 {
		t.Fatalf("SortKeyPointsByResponse wrong: %+v", kps)
	}
	if got := FilterByResponse(kps, 5); len(got) != 2 {
		t.Fatalf("FilterByResponse got %d want 2", len(got))
	}
	border := FilterByBorder(kps, 10, 10, 2)
	for _, k := range border {
		if k.Pt.X < 2 || k.Pt.X > 7 {
			t.Fatalf("FilterByBorder kept edge point %+v", k)
		}
	}
	pts := ConvertKeyPointsToPoints(kps)
	back := ConvertPointsToKeyPoints(pts, 7)
	if len(back) != len(kps) {
		t.Fatalf("conversion length mismatch")
	}
	if back[0].Point() != pts[0] {
		t.Fatalf("round-trip point mismatch")
	}
}
