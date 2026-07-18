package tracking

import "sort"

// Track is one identity maintained by an [IoUTracker]: a stable ID, its most
// recent bounding box and lifecycle counters.
type Track struct {
	// ID is the stable identifier assigned when the track was created.
	ID int
	// Box is the most recent bounding box associated with the track.
	Box Rect
	// Hits is the total number of detections matched to this track.
	Hits int
	// Age is the number of frames since the track was created.
	Age int
	// TimeSinceUpdate is the number of frames since the track last matched a
	// detection; a freshly matched track has zero.
	TimeSinceUpdate int
}

// IoUTracker associates detections with tracks across frames by
// intersection-over-union overlap, the core of the classic "IOU Tracker". Each
// frame's detection boxes are matched greedily to existing tracks by descending
// IoU above a threshold; matched tracks are updated, unmatched tracks age and
// are dropped once stale, and unmatched detections start new tracks. The
// association is deterministic.
type IoUTracker struct {
	nextID       int
	tracks       map[int]*Track
	iouThreshold float64
	maxAge       int
}

// NewIoUTracker creates a tracker. iouThreshold is the minimum overlap for a
// detection to match a track (a value in (0, 1]); maxAge is the number of
// consecutive frames a track may go unmatched before it is removed.
func NewIoUTracker(iouThreshold float64, maxAge int) *IoUTracker {
	return &IoUTracker{
		nextID:       0,
		tracks:       make(map[int]*Track),
		iouThreshold: iouThreshold,
		maxAge:       maxAge,
	}
}

// Count returns the number of live tracks.
func (t *IoUTracker) Count() int { return len(t.tracks) }

// create starts a new track from a detection box.
func (t *IoUTracker) create(box Rect) {
	t.tracks[t.nextID] = &Track{ID: t.nextID, Box: box, Hits: 1}
	t.nextID++
}

// Update associates the detection boxes of the current frame with existing
// tracks by IoU and returns a snapshot of the live tracks ordered by ID. Tracks
// unmatched for more than maxAge frames are removed.
func (t *IoUTracker) Update(detections []Rect) []Track {
	ids := make([]int, 0, len(t.tracks))
	for id := range t.tracks {
		ids = append(ids, id)
	}
	sort.Ints(ids)

	type pair struct {
		trkIdx int
		detIdx int
		iou    float64
	}
	var pairs []pair
	for ti, id := range ids {
		tb := t.tracks[id].Box
		for di, d := range detections {
			iou := tb.IoU(d)
			if iou >= t.iouThreshold {
				pairs = append(pairs, pair{ti, di, iou})
			}
		}
	}
	// Greedy assignment by descending IoU; ties broken by indices.
	sort.Slice(pairs, func(a, b int) bool {
		if pairs[a].iou != pairs[b].iou {
			return pairs[a].iou > pairs[b].iou
		}
		if pairs[a].trkIdx != pairs[b].trkIdx {
			return pairs[a].trkIdx < pairs[b].trkIdx
		}
		return pairs[a].detIdx < pairs[b].detIdx
	})

	usedTrk := make([]bool, len(ids))
	usedDet := make([]bool, len(detections))
	for _, p := range pairs {
		if usedTrk[p.trkIdx] || usedDet[p.detIdx] {
			continue
		}
		usedTrk[p.trkIdx] = true
		usedDet[p.detIdx] = true
		trk := t.tracks[ids[p.trkIdx]]
		trk.Box = detections[p.detIdx]
		trk.Hits++
		trk.Age++
		trk.TimeSinceUpdate = 0
	}

	// Age unmatched tracks and drop the stale ones.
	for ti, id := range ids {
		if usedTrk[ti] {
			continue
		}
		trk := t.tracks[id]
		trk.Age++
		trk.TimeSinceUpdate++
		if trk.TimeSinceUpdate > t.maxAge {
			delete(t.tracks, id)
		}
	}
	// Create tracks for unmatched detections.
	for di, d := range detections {
		if !usedDet[di] {
			t.create(d)
		}
	}
	return t.Tracks()
}

// Tracks returns a snapshot of the live tracks ordered by ID.
func (t *IoUTracker) Tracks() []Track {
	ids := make([]int, 0, len(t.tracks))
	for id := range t.tracks {
		ids = append(ids, id)
	}
	sort.Ints(ids)
	out := make([]Track, 0, len(ids))
	for _, id := range ids {
		out = append(out, *t.tracks[id])
	}
	return out
}
