package textdet

import (
	"testing"

	cv "github.com/malcolmston/opencv"
)

func TestHorizontalProjection(t *testing.T) {
	// Two full-width bars: rows 2..4 and rows 8..10 of a 12x6 image.
	m := cv.NewMat(12, 6, 1)
	paintRect(m, 0, 2, 6, 3, 255)
	paintRect(m, 0, 8, 6, 3, 255)
	prof, err := HorizontalProjection(m)
	if err != nil {
		t.Fatal(err)
	}
	want := []int{0, 0, 6, 6, 6, 0, 0, 0, 6, 6, 6, 0}
	for i := range want {
		if prof[i] != want[i] {
			t.Fatalf("HorizontalProjection = %v, want %v", prof, want)
		}
	}
}

func TestVerticalProjection(t *testing.T) {
	m := cv.NewMat(3, 6, 1)
	// Ink columns 1..2 and 4..5.
	paintRect(m, 1, 0, 2, 3, 255)
	paintRect(m, 4, 0, 2, 3, 255)
	prof, err := VerticalProjection(m)
	if err != nil {
		t.Fatal(err)
	}
	want := []int{0, 3, 3, 0, 3, 3}
	for i := range want {
		if prof[i] != want[i] {
			t.Fatalf("VerticalProjection = %v, want %v", prof, want)
		}
	}
}

func TestSegmentBandsMergeAndDrop(t *testing.T) {
	prof := []int{0, 5, 5, 0, 5, 0, 0, 9, 0}
	// threshold 0, minGap 2 (merge runs 1..4 across the single-zero gap),
	// minLength 2 (drop the single-column run at index 7).
	bands, err := SegmentBands(prof, 0, 2, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(bands) != 1 {
		t.Fatalf("bands = %+v, want 1 merged band", bands)
	}
	if bands[0].Start != 1 || bands[0].End != 4 {
		t.Fatalf("band = %+v, want [1,4]", bands[0])
	}
}

func TestSegmentTextLines(t *testing.T) {
	m := cv.NewMat(12, 6, 1)
	paintRect(m, 0, 2, 6, 3, 255)
	paintRect(m, 0, 8, 6, 3, 255)
	lines, err := SegmentTextLines(m, 0, 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(lines) != 2 {
		t.Fatalf("lines = %d, want 2", len(lines))
	}
	if lines[0].Y != 2 || lines[0].Height != 3 || lines[0].Width != 6 {
		t.Fatalf("line0 = %+v, want Y=2 H=3 W=6", lines[0])
	}
	if lines[1].Y != 8 || lines[1].Height != 3 {
		t.Fatalf("line1 = %+v, want Y=8 H=3", lines[1])
	}
}

func TestSegmentWordsAndCharacters(t *testing.T) {
	// One line, two words each of two ink columns with a 1-col gap inside a
	// word and a 2-col gap between words.
	m := cv.NewMat(3, 10, 1)
	paintRect(m, 0, 0, 1, 3, 255) // col 0
	paintRect(m, 2, 0, 1, 3, 255) // col 2 (1-col intra gap at col 1)
	paintRect(m, 5, 0, 1, 3, 255) // col 5
	paintRect(m, 7, 0, 1, 3, 255) // col 7
	// Words: merge gaps < 2 -> the intra-word single-column gap merges.
	words, err := SegmentWords(m, 0, 2, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(words) != 2 {
		t.Fatalf("words = %d, want 2", len(words))
	}
	if words[0].X != 0 || words[0].Width != 3 {
		t.Fatalf("word0 = %+v, want X=0 W=3", words[0])
	}
	// Characters: no gap merging -> four separate strokes.
	chars, err := SegmentCharacters(m, 0, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(chars) != 4 {
		t.Fatalf("chars = %d, want 4", len(chars))
	}
}

func TestEstimateSkewHorizontal(t *testing.T) {
	// A horizontal bar has maximum projection variance at angle 0.
	m := cv.NewMat(21, 21, 1)
	paintRect(m, 2, 9, 17, 3, 255)
	angle, err := EstimateSkew(m, 8, 1)
	if err != nil {
		t.Fatal(err)
	}
	if angle != 0 {
		t.Fatalf("EstimateSkew = %v, want 0", angle)
	}
}

func TestCorrectSkewRoundTrip(t *testing.T) {
	m := cv.NewMat(21, 21, 1)
	paintRect(m, 2, 9, 17, 3, 255)
	out, angle, err := CorrectSkew(m, 8, 1)
	if err != nil {
		t.Fatal(err)
	}
	if angle != 0 {
		t.Fatalf("CorrectSkew angle = %v, want 0", angle)
	}
	if out.Rows != m.Rows || out.Cols != m.Cols {
		t.Fatalf("CorrectSkew changed dimensions")
	}
}

func TestProjectionVarianceMonotone(t *testing.T) {
	// Variance at 0 degrees should exceed variance at a large tilt for a
	// horizontal bar.
	m := cv.NewMat(21, 21, 1)
	paintRect(m, 2, 9, 17, 3, 255)
	v0, _ := ProjectionVariance(m, 0)
	v30, _ := ProjectionVariance(m, 30)
	if !(v0 > v30) {
		t.Fatalf("variance at 0 (%v) not greater than at 30 (%v)", v0, v30)
	}
}

func TestProjectionEmpty(t *testing.T) {
	var empty cv.Mat
	if _, err := HorizontalProjection(&empty); err != ErrEmpty {
		t.Fatalf("empty err = %v, want ErrEmpty", err)
	}
	if _, err := EstimateSkew(newGray(4, 4, 0), 0, 1); err != ErrInvalidArgument {
		t.Fatalf("bad maxAngle err = %v, want ErrInvalidArgument", err)
	}
}
