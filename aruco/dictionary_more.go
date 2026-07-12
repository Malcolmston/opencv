package aruco

import "sync"

// This file adds larger predefined dictionaries and a general dictionary
// generator, complementing the 4x4 and 5x5 families in dictionary.go. Because
// [GetPredefinedDictionary] is closed over a fixed set of names, the 6x6
// families are exposed through their own accessors rather than by extending it.

// Predefined dictionary names for the 6x6 families provided by this file. They
// continue the numbering of [Dict4x4] and [Dict5x5] and are accepted by
// [GetPredefinedDictionary6x6].
const (
	// Dict6x6 is a dictionary of markers with a 6x6 inner grid (36 bits) holding
	// 250 identifiers, mirroring OpenCV's DICT_6X6_250.
	Dict6x6 PredefinedDictionaryName = 2 + iota
	// Dict6x6Small is a smaller 6x6 family of 100 identifiers with a larger
	// inter-marker distance, mirroring OpenCV's DICT_6X6_100.
	Dict6x6Small
)

var (
	dict6x6Once   sync.Once
	dict6x6       *Dictionary
	dict6x6SmOnce sync.Once
	dict6x6Sm     *Dictionary
)

// GetPredefinedDictionary6x6 returns one of the 6x6 predefined dictionaries
// ([Dict6x6] or [Dict6x6Small]). The dictionary is generated once and cached, so
// repeated calls return the same pointer. It panics on any other name.
//
// The 6x6 grid packs 36 bits, which supports many more well-separated markers
// than the 4x4 and 5x5 families and tolerates more bit errors per reading. Use
// it when a scene needs a large number of distinct markers or extra robustness.
func GetPredefinedDictionary6x6(name PredefinedDictionaryName) *Dictionary {
	switch name {
	case Dict6x6:
		dict6x6Once.Do(func() {
			dict6x6 = buildDictionary("DICT_6X6_250", 6, 250, 8, 0x2545f4914f6cdd1d)
		})
		return dict6x6
	case Dict6x6Small:
		dict6x6SmOnce.Do(func() {
			dict6x6Sm = buildDictionary("DICT_6X6_100", 6, 100, 10, 0x1d872b41a1e4c5f9)
		})
		return dict6x6Sm
	default:
		panic("aruco: GetPredefinedDictionary6x6 unknown dictionary name")
	}
}

// GenerateCustomDictionary deterministically builds a fresh dictionary of up to
// count markers with a bitsPerSide-by-bitsPerSide inner grid, where accepted
// markers (and all of their 90-degree rotations) are mutually at least
// minDistance cells apart in Hamming distance. seed selects the deterministic
// marker family: the same arguments always yield the same dictionary, and
// different seeds yield different but equally valid families. This mirrors
// OpenCV's cv::aruco::extendDictionary / custom-dictionary generation.
//
// The match tolerance is floor((minDistance-1)/2), so a reading can never fall
// within tolerance of two different markers. It panics if bitsPerSide is below
// 4, count is not positive, or minDistance is below 1. The returned dictionary
// may hold fewer than count markers if the constraints leave no more room; its
// Name is "DICT_CUSTOM".
func GenerateCustomDictionary(bitsPerSide, count, minDistance int, seed uint64) *Dictionary {
	if bitsPerSide < 4 {
		panic("aruco: GenerateCustomDictionary requires bitsPerSide >= 4")
	}
	if count <= 0 {
		panic("aruco: GenerateCustomDictionary requires a positive count")
	}
	if minDistance < 1 {
		panic("aruco: GenerateCustomDictionary requires minDistance >= 1")
	}
	return buildDictionary("DICT_CUSTOM", bitsPerSide, count, minDistance, seed)
}
