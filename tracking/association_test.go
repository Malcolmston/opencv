package tracking

import "testing"

func TestCentroidTrackerAssignsStableIDs(t *testing.T) {
	ct := NewCentroidTracker(3, 50)
	// Two objects appear.
	objs := ct.Update([]Rect{NewRect(0, 0, 10, 10), NewRect(100, 100, 10, 10)})
	requireTrue(t, len(objs) == 2, "expected 2 objects, got %d", len(objs))
	id0, id1 := objs[0].ID, objs[1].ID

	// Both move slightly; IDs must be preserved by nearest-centroid matching.
	objs = ct.Update([]Rect{NewRect(2, 1, 10, 10), NewRect(103, 102, 10, 10)})
	requireTrue(t, objs[0].ID == id0, "first object changed ID")
	requireTrue(t, objs[1].ID == id1, "second object changed ID")
	requireTrue(t, approx(objs[0].Centroid.X, 7, 0.001), "centroid X updated wrong")
}

func TestCentroidTrackerDisappearance(t *testing.T) {
	ct := NewCentroidTracker(2, 50)
	ct.Update([]Rect{NewRect(0, 0, 10, 10)})
	// No detections for three frames; maxDisappeared is 2, so it should be gone.
	ct.Update(nil)
	requireTrue(t, ct.Count() == 1, "object should survive 1 missed frame")
	ct.Update(nil)
	requireTrue(t, ct.Count() == 1, "object should survive 2 missed frames")
	ct.Update(nil)
	requireTrue(t, ct.Count() == 0, "object should be removed after 3 missed frames")
}

func TestCentroidTrackerRegistersNew(t *testing.T) {
	ct := NewCentroidTracker(3, 20)
	ct.Update([]Rect{NewRect(0, 0, 10, 10)})
	// A far-away detection is beyond maxDistance, so a new ID is created and the
	// old object ages.
	objs := ct.Update([]Rect{NewRect(200, 200, 10, 10)})
	requireTrue(t, ct.Count() == 2, "expected 2 tracked objects, got %d", ct.Count())
	_ = objs
}

func TestIoUTrackerAssociates(t *testing.T) {
	it := NewIoUTracker(0.3, 2)
	tracks := it.Update([]Rect{NewRect(0, 0, 20, 20), NewRect(100, 100, 20, 20)})
	requireTrue(t, len(tracks) == 2, "expected 2 tracks")
	id0 := tracks[0].ID

	// Overlapping detection continues the first track (IoU well above 0.3).
	tracks = it.Update([]Rect{NewRect(2, 2, 20, 20)})
	requireTrue(t, tracks[0].ID == id0 || (len(tracks) > 1 && tracks[1].ID == id0), "track 0 should persist by IoU")

	var found *Track
	for i := range tracks {
		if tracks[i].ID == id0 {
			found = &tracks[i]
		}
	}
	requireTrue(t, found != nil, "track id0 not found")
	requireTrue(t, found.Hits == 2, "track should have 2 hits, got %d", found.Hits)
	requireTrue(t, found.TimeSinceUpdate == 0, "matched track should have TimeSinceUpdate 0")
}

func TestIoUTrackerAges(t *testing.T) {
	it := NewIoUTracker(0.3, 1)
	it.Update([]Rect{NewRect(0, 0, 20, 20)})
	// No overlap: the track goes unmatched.
	it.Update([]Rect{NewRect(500, 500, 20, 20)})
	requireTrue(t, it.Count() == 2, "old track should still exist after 1 miss")
	// Second miss exceeds maxAge=1, dropping the original track.
	it.Update([]Rect{NewRect(500, 500, 20, 20)})
	requireTrue(t, it.Count() == 1, "stale track should be removed, count = %d", it.Count())
}
