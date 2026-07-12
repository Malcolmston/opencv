package aruco

import "sync"

// Dictionary is a family of markers that share a fixed inner grid size. Each
// marker stores its identifier together with its bit grid precomputed in all
// four 90-degree rotations, so that detection can match a reading against every
// orientation cheaply. Construct dictionaries with [GetPredefinedDictionary];
// the zero value is not usable.
type Dictionary struct {
	// Name is a human-readable label such as "DICT_4X4".
	Name string
	// bitsPerSide is the number of inner cells along one side (4 or 5). The
	// full marker, including its black border, is bitsPerSide+2 cells wide.
	bitsPerSide int
	// markers holds every marker, indexed by dictionary position (not id).
	markers []dictMarker
	// tolerance is the maximum Hamming distance, in cells, at which a reading
	// is still accepted as a match. It is derived from the dictionary's minimum
	// inter-marker distance so that a match is unambiguous.
	tolerance int
}

// dictMarker is a single dictionary entry: an identifier and its inner grid in
// all four rotations. Each rotation is a flat bitsPerSide*bitsPerSide slice in
// row-major order with 1 for white and 0 for black.
type dictMarker struct {
	id   int
	rots [4][]byte
}

// BitsPerSide returns the number of inner cells along one side of a marker in
// this dictionary. The rendered marker, including its black border, is
// BitsPerSide()+2 cells wide.
func (d *Dictionary) BitsPerSide() int { return d.bitsPerSide }

// Size returns the number of distinct markers (identifiers) in the dictionary.
// Valid identifiers are 0 through Size()-1.
func (d *Dictionary) Size() int { return len(d.markers) }

// Tolerance returns the maximum Hamming distance, in cells, at which a cell
// reading is still accepted as a match for a dictionary marker.
func (d *Dictionary) Tolerance() int { return d.tolerance }

// bits returns the canonical (unrotated) inner grid of the marker with the
// given id, or nil if the id is out of range.
func (d *Dictionary) bits(id int) []byte {
	if id < 0 || id >= len(d.markers) {
		return nil
	}
	return d.markers[id].rots[0]
}

// PredefinedDictionaryName selects one of the dictionaries shipped with the
// package for [GetPredefinedDictionary].
type PredefinedDictionaryName int

const (
	// Dict4x4 is a dictionary of markers with a 4x4 inner grid (16 bits).
	Dict4x4 PredefinedDictionaryName = iota
	// Dict5x5 is a dictionary of markers with a 5x5 inner grid (25 bits).
	Dict5x5
)

var (
	dict4x4Once sync.Once
	dict4x4     *Dictionary
	dict5x5Once sync.Once
	dict5x5     *Dictionary
)

// GetPredefinedDictionary returns the requested predefined dictionary. The
// dictionary is generated once and cached, so repeated calls return the same
// pointer. It panics on an unknown name.
func GetPredefinedDictionary(name PredefinedDictionaryName) *Dictionary {
	switch name {
	case Dict4x4:
		dict4x4Once.Do(func() {
			dict4x4 = buildDictionary("DICT_4X4", 4, 50, 3, 0x9e3779b97f4a7c15)
		})
		return dict4x4
	case Dict5x5:
		dict5x5Once.Do(func() {
			dict5x5 = buildDictionary("DICT_5X5", 5, 50, 5, 0xd1b54a32d192ed03)
		})
		return dict5x5
	default:
		panic("aruco: GetPredefinedDictionary unknown dictionary name")
	}
}

// buildDictionary deterministically generates up to want markers with the given
// inner grid side. A candidate grid is accepted only when its four rotations are
// mutually at least minDist cells apart (so its orientation is recoverable) and
// it is at least minDist cells from every rotation of every already-accepted
// marker (so identifiers do not alias). Generation is driven by a fixed-seed
// linear congruential generator, which makes the resulting set stable across
// runs. The match tolerance is set to floor((minDist-1)/2) so that a reading can
// never lie within tolerance of two different markers.
func buildDictionary(name string, side, want, minDist int, seed uint64) *Dictionary {
	n := side * side
	rng := seed
	next := func() uint64 {
		// SplitMix64: a fast, well-distributed deterministic generator.
		rng += 0x9e3779b97f4a7c15
		z := rng
		z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
		z = (z ^ (z >> 27)) * 0x94d049bb133111eb
		return z ^ (z >> 31)
	}

	d := &Dictionary{Name: name, bitsPerSide: side, tolerance: (minDist - 1) / 2}
	const maxAttempts = 1 << 21
	for attempt := 0; attempt < maxAttempts && len(d.markers) < want; attempt++ {
		g := make([]byte, n)
		var ones int
		for i := 0; i < n; i++ {
			if next()&1 == 1 {
				g[i] = 1
				ones++
			}
		}
		// Reject nearly-blank or nearly-full grids: they carry little signal.
		if ones < n/4 || ones > 3*n/4 {
			continue
		}
		rots := fourRotations(g, side)
		if !rotationsSeparated(rots, minDist) {
			continue
		}
		if !separatedFromAll(d.markers, rots, minDist) {
			continue
		}
		d.markers = append(d.markers, dictMarker{id: len(d.markers), rots: rots})
	}
	return d
}

// fourRotations returns g and its 90/180/270-degree clockwise rotations, each a
// fresh flat slice.
func fourRotations(g []byte, side int) [4][]byte {
	var rots [4][]byte
	rots[0] = g
	for k := 1; k < 4; k++ {
		rots[k] = rotateGridCW(rots[k-1], side)
	}
	return rots
}

// rotateGridCW returns a new grid that is g rotated 90 degrees clockwise. For a
// side*side grid the output cell (i, j) is taken from input cell (side-1-j, i).
func rotateGridCW(g []byte, side int) []byte {
	out := make([]byte, len(g))
	for i := 0; i < side; i++ {
		for j := 0; j < side; j++ {
			out[i*side+j] = g[(side-1-j)*side+i]
		}
	}
	return out
}

// rotationsSeparated reports whether all four rotations of a marker are pairwise
// at least minDist cells apart, which guarantees its orientation can be told
// apart from a noisy reading.
func rotationsSeparated(rots [4][]byte, minDist int) bool {
	for a := 0; a < 4; a++ {
		for b := a + 1; b < 4; b++ {
			if hamming(rots[a], rots[b]) < minDist {
				return false
			}
		}
	}
	return true
}

// separatedFromAll reports whether the candidate rotations are at least minDist
// cells from every rotation of every marker already in the dictionary.
func separatedFromAll(markers []dictMarker, cand [4][]byte, minDist int) bool {
	for i := range markers {
		for a := 0; a < 4; a++ {
			for b := 0; b < 4; b++ {
				if hamming(markers[i].rots[a], cand[b]) < minDist {
					return false
				}
			}
		}
	}
	return true
}

// hamming returns the number of positions at which the two equal-length bit
// slices differ.
func hamming(a, b []byte) int {
	d := 0
	for i := range a {
		if a[i] != b[i] {
			d++
		}
	}
	return d
}

// matchGrid compares a read inner grid against every marker under all four
// rotations. It returns the best-matching id and the rotation index k such that
// the reading equals the marker rotated clockwise k times, or ok=false when no
// marker lies within the dictionary tolerance. Ties in distance are resolved in
// favour of the smaller distance; a genuine tie between two different ids is
// rejected as ambiguous.
func matchGrid(d *Dictionary, read []byte) (id, k int, ok bool) {
	bestDist := d.tolerance + 1
	bestID, bestK := -1, 0
	ambiguous := false
	for mi := range d.markers {
		for r := 0; r < 4; r++ {
			dist := hamming(read, d.markers[mi].rots[r])
			if dist < bestDist {
				bestDist = dist
				bestID = d.markers[mi].id
				bestK = r
				ambiguous = false
			} else if dist == bestDist && d.markers[mi].id != bestID {
				ambiguous = true
			}
		}
	}
	if bestID < 0 || ambiguous {
		return 0, 0, false
	}
	return bestID, bestK, true
}
