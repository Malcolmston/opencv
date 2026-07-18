package template2

import "testing"

func TestNonMaxSuppressionOverlap(t *testing.T) {
	matches := []Match{
		{X: 0, Y: 0, Width: 4, Height: 4, Score: 0.9},
		{X: 1, Y: 1, Width: 4, Height: 4, Score: 0.8}, // heavy overlap with the first
		{X: 20, Y: 20, Width: 4, Height: 4, Score: 0.7},
	}
	kept := NonMaxSuppression(matches, 0.3, true)
	if len(kept) != 2 {
		t.Fatalf("expected 2 kept matches, got %d", len(kept))
	}
	if kept[0].Score != 0.9 {
		t.Fatalf("expected best match first, got %g", kept[0].Score)
	}
	// The distant box survives.
	if kept[1].X != 20 {
		t.Fatalf("expected distant box kept, got X=%d", kept[1].X)
	}
}

func TestNonMaxSuppressionKeepsAllWhenDisjoint(t *testing.T) {
	matches := []Match{
		{X: 0, Y: 0, Width: 2, Height: 2, Score: 0.9},
		{X: 5, Y: 5, Width: 2, Height: 2, Score: 0.8},
		{X: 10, Y: 10, Width: 2, Height: 2, Score: 0.7},
	}
	kept := NonMaxSuppression(matches, 0.3, true)
	if len(kept) != 3 {
		t.Fatalf("expected all 3 kept, got %d", len(kept))
	}
}

func TestNonMaxSuppressionLowerBetter(t *testing.T) {
	// For SSD-style scores, the lowest score is the best and should be kept.
	matches := []Match{
		{X: 0, Y: 0, Width: 4, Height: 4, Score: 100},
		{X: 1, Y: 1, Width: 4, Height: 4, Score: 5}, // best, overlaps the first
	}
	kept := NonMaxSuppression(matches, 0.3, false)
	if len(kept) != 1 || kept[0].Score != 5 {
		t.Fatalf("expected single best (5), got %+v", kept)
	}
}

func TestNonMaxSuppressionDistance(t *testing.T) {
	matches := []Match{
		{X: 0, Y: 0, Width: 2, Height: 2, Score: 0.9},
		{X: 1, Y: 0, Width: 2, Height: 2, Score: 0.8}, // center 1 px away
		{X: 10, Y: 0, Width: 2, Height: 2, Score: 0.7},
	}
	kept := NonMaxSuppressionDistance(matches, 3, true)
	if len(kept) != 2 {
		t.Fatalf("expected 2 kept, got %d", len(kept))
	}
}

func TestFilterAndSort(t *testing.T) {
	matches := []Match{
		{Score: 0.5}, {Score: 0.9}, {Score: 0.2},
	}
	SortByScore(matches, true)
	if matches[0].Score != 0.9 || matches[2].Score != 0.2 {
		t.Fatalf("sort descending failed: %+v", matches)
	}
	kept := FilterByScore(matches, 0.4, true)
	if len(kept) != 2 {
		t.Fatalf("expected 2 above 0.4, got %d", len(kept))
	}
}
