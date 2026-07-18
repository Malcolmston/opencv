package tracking

import (
	"math"
	"sort"
)

// TrackedObject is one identity maintained by a [CentroidTracker]: a stable
// integer ID, the current centroid and detection box, and how many consecutive
// frames the object has gone unmatched.
type TrackedObject struct {
	// ID is the stable identifier assigned when the object first appeared.
	ID int
	// Centroid is the current object centre.
	Centroid Point2f
	// Box is the most recent detection box associated with the object.
	Box Rect
	// Disappeared is the number of consecutive frames the object has been
	// unmatched; it is reset to zero on every match.
	Disappeared int
}

// CentroidTracker assigns stable identities to detections across frames by
// nearest-centroid association. Each frame's detection boxes are matched to
// existing objects greedily by ascending centroid distance; unmatched objects
// accumulate a disappearance count and are dropped once it exceeds a limit, and
// unmatched detections spawn new identities. The association is deterministic.
type CentroidTracker struct {
	nextID         int
	objects        map[int]*TrackedObject
	maxDisappeared int
	maxDistance    float64
}

// NewCentroidTracker creates a tracker. maxDisappeared is the number of
// consecutive unmatched frames an object may survive before it is removed.
// maxDistance is the largest centroid distance (in pixels) allowed for a match;
// pass 0 or a negative value to disable the distance gate.
func NewCentroidTracker(maxDisappeared int, maxDistance float64) *CentroidTracker {
	return &CentroidTracker{
		nextID:         0,
		objects:        make(map[int]*TrackedObject),
		maxDisappeared: maxDisappeared,
		maxDistance:    maxDistance,
	}
}

// Count returns the number of currently tracked objects.
func (c *CentroidTracker) Count() int { return len(c.objects) }

// register adds a new object for the given detection.
func (c *CentroidTracker) register(box Rect) {
	c.objects[c.nextID] = &TrackedObject{ID: c.nextID, Centroid: box.Center(), Box: box}
	c.nextID++
}

// Update associates the detection boxes of the current frame with existing
// identities and returns a snapshot of all live objects, ordered by ID. Objects
// unmatched for more than maxDisappeared frames are removed.
func (c *CentroidTracker) Update(detections []Rect) []TrackedObject {
	if len(detections) == 0 {
		// No detections: age every object and drop the stale ones.
		for id, obj := range c.objects {
			obj.Disappeared++
			if obj.Disappeared > c.maxDisappeared {
				delete(c.objects, id)
			}
		}
		return c.Objects()
	}

	if len(c.objects) == 0 {
		for _, d := range detections {
			c.register(d)
		}
		return c.Objects()
	}

	// Existing object IDs in a stable order.
	ids := make([]int, 0, len(c.objects))
	for id := range c.objects {
		ids = append(ids, id)
	}
	sort.Ints(ids)

	type pair struct {
		objIdx int
		detIdx int
		dist   float64
	}
	var pairs []pair
	for oi, id := range ids {
		oc := c.objects[id].Centroid
		for di, d := range detections {
			dist := oc.Distance(d.Center())
			if c.maxDistance > 0 && dist > c.maxDistance {
				continue
			}
			pairs = append(pairs, pair{oi, di, dist})
		}
	}
	// Greedy assignment by ascending distance; ties broken by indices for
	// determinism.
	sort.Slice(pairs, func(a, b int) bool {
		if pairs[a].dist != pairs[b].dist {
			return pairs[a].dist < pairs[b].dist
		}
		if pairs[a].objIdx != pairs[b].objIdx {
			return pairs[a].objIdx < pairs[b].objIdx
		}
		return pairs[a].detIdx < pairs[b].detIdx
	})

	usedObj := make([]bool, len(ids))
	usedDet := make([]bool, len(detections))
	for _, p := range pairs {
		if usedObj[p.objIdx] || usedDet[p.detIdx] {
			continue
		}
		usedObj[p.objIdx] = true
		usedDet[p.detIdx] = true
		obj := c.objects[ids[p.objIdx]]
		d := detections[p.detIdx]
		obj.Centroid = d.Center()
		obj.Box = d
		obj.Disappeared = 0
	}

	// Age unmatched objects.
	for oi, id := range ids {
		if usedObj[oi] {
			continue
		}
		obj := c.objects[id]
		obj.Disappeared++
		if obj.Disappeared > c.maxDisappeared {
			delete(c.objects, id)
		}
	}
	// Register unmatched detections.
	for di, d := range detections {
		if !usedDet[di] {
			c.register(d)
		}
	}
	return c.Objects()
}

// Objects returns a snapshot of the tracked objects ordered by ID.
func (c *CentroidTracker) Objects() []TrackedObject {
	ids := make([]int, 0, len(c.objects))
	for id := range c.objects {
		ids = append(ids, id)
	}
	sort.Ints(ids)
	out := make([]TrackedObject, 0, len(ids))
	for _, id := range ids {
		out = append(out, *c.objects[id])
	}
	return out
}

// euclidean is a small helper kept for clarity in tests and internal use.
func euclidean(a, b Point2f) float64 { return math.Hypot(a.X-b.X, a.Y-b.Y) }
