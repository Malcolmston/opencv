package xfeatures2d

import "math"

// DMatch is a correspondence between a descriptor in a query set and a
// descriptor in a train set, mirroring OpenCV's cv::DMatch. Distance is the
// descriptor dissimilarity ([HammingDistance] for binary descriptors,
// [L2Distance] for float descriptors).
type DMatch struct {
	QueryIdx int
	TrainIdx int
	Distance float64
}

// MatchBruteForceHamming returns, for every query descriptor, its nearest train
// descriptor by Hamming distance (a brute-force cross-check-free matcher). The
// returned matches are in query order. All descriptors must have the same byte
// length. An empty train set yields no matches.
func MatchBruteForceHamming(query, train [][]byte) []DMatch {
	if len(train) == 0 {
		return nil
	}
	matches := make([]DMatch, 0, len(query))
	for qi, q := range query {
		best := math.MaxInt
		bestJ := -1
		for tj, t := range train {
			d := HammingDistance(q, t)
			if d < best {
				best = d
				bestJ = tj
			}
		}
		matches = append(matches, DMatch{QueryIdx: qi, TrainIdx: bestJ, Distance: float64(best)})
	}
	return matches
}

// MatchBruteForceL2 returns, for every query descriptor, its nearest train
// descriptor by L2 distance. The returned matches are in query order. All
// descriptors must have the same length. An empty train set yields no matches.
func MatchBruteForceL2(query, train [][]float64) []DMatch {
	if len(train) == 0 {
		return nil
	}
	matches := make([]DMatch, 0, len(query))
	for qi, q := range query {
		best := math.Inf(1)
		bestJ := -1
		for tj, t := range train {
			d := L2Distance(q, t)
			if d < best {
				best = d
				bestJ = tj
			}
		}
		matches = append(matches, DMatch{QueryIdx: qi, TrainIdx: bestJ, Distance: best})
	}
	return matches
}

// gmsGrid is the number of cells per image side used by the GMS grid.
const gmsGrid = 20

// cellCoord returns the (cx, cy) grid cell of a point in an image of the given
// size split into gx×gy cells.
func cellCoord(x, y, rows, cols, gx, gy int) (int, int) {
	cx := x * gx / cols
	cy := y * gy / rows
	if cx >= gx {
		cx = gx - 1
	}
	if cy >= gy {
		cy = gy - 1
	}
	if cx < 0 {
		cx = 0
	}
	if cy < 0 {
		cy = 0
	}
	return cx, cy
}

// offset is a 3×3 neighbourhood displacement in grid-cell units.
type offset struct{ dx, dy int }

var gmsOffsets = []offset{
	{-1, -1}, {0, -1}, {1, -1},
	{-1, 0}, {0, 0}, {1, 0},
	{-1, 1}, {0, 1}, {1, 1},
}

// gmsRotations lists the neighbourhood transforms tried when rotation
// robustness is enabled: the four 90° rotations, each with and without a
// horizontal mirror (eight in total).
var gmsRotations = []func(offset) offset{
	func(o offset) offset { return o },
	func(o offset) offset { return offset{o.dy, -o.dx} },
	func(o offset) offset { return offset{-o.dx, -o.dy} },
	func(o offset) offset { return offset{-o.dy, o.dx} },
	func(o offset) offset { return offset{-o.dx, o.dy} },
	func(o offset) offset { return offset{o.dy, o.dx} },
	func(o offset) offset { return offset{o.dx, -o.dy} },
	func(o offset) offset { return offset{-o.dy, -o.dx} },
}

// MatchGMS filters putative matches with Grid-based Motion Statistics, a port of
// OpenCV's cv::xfeatures2d::matchGMS.
//
// GMS rests on the observation that a correct match is supported by many other
// matches in its immediate neighbourhood, whereas false matches are
// statistically isolated. Both images are partitioned into a grid; matches are
// bucketed by the cell-pair (query cell, train cell) they connect; and a
// cell-pair is accepted when the number of matches in its 3×3 cell neighbourhood
// exceeds a statistical threshold thresholdFactor·sqrt(n). All matches inside an
// accepted cell-pair are returned as inliers. When withRotation is set the eight
// canonical neighbourhood rotations are tried and the best support is used; when
// withScale is set several grid resolutions are combined. rows/cols give the two
// image sizes; kp1/kp2 are the keypoints the matches index; thresholdFactor is
// typically about 6.
func MatchGMS(rows1, cols1, rows2, cols2 int, kp1, kp2 []KeyPoint, matches []DMatch, withRotation, withScale bool, thresholdFactor float64) []DMatch {
	if len(matches) == 0 {
		return nil
	}
	scales := []float64{1.0}
	if withScale {
		scales = []float64{1.0, 0.5, 2.0}
	}
	accepted := make([]bool, len(matches))
	for _, sc := range scales {
		runGMS(rows1, cols1, rows2, cols2, kp1, kp2, matches, withRotation, thresholdFactor, sc, accepted)
	}
	var out []DMatch
	for i, ok := range accepted {
		if ok {
			out = append(out, matches[i])
		}
	}
	return out
}

// runGMS evaluates the grid statistics at one image-2 scale and marks accepted
// matches in the shared accepted slice.
func runGMS(rows1, cols1, rows2, cols2 int, kp1, kp2 []KeyPoint, matches []DMatch, withRotation bool, factor, scale float64, accepted []bool) {
	gx2 := int(math.Round(float64(gmsGrid) * scale))
	gy2 := gx2
	if gx2 < 1 {
		gx2 = 1
		gy2 = 1
	}

	type cellPair struct{ c1x, c1y, c2x, c2y int }
	cells := make([]cellPair, len(matches))
	counts := make(map[[4]int]int)
	members := make(map[[4]int][]int)
	for i, m := range matches {
		q := kp1[m.QueryIdx].Pt
		t := kp2[m.TrainIdx].Pt
		c1x, c1y := cellCoord(q.X, q.Y, rows1, cols1, gmsGrid, gmsGrid)
		c2x, c2y := cellCoord(t.X, t.Y, rows2, cols2, gx2, gy2)
		cells[i] = cellPair{c1x, c1y, c2x, c2y}
		key := [4]int{c1x, c1y, c2x, c2y}
		counts[key]++
		members[key] = append(members[key], i)
	}

	rotations := gmsRotations[:1]
	if withRotation {
		rotations = gmsRotations
	}

	for key, memb := range members {
		bestS := 0
		for _, rot := range rotations {
			S := 0
			for _, o := range gmsOffsets {
				o2 := rot(o)
				n1x, n1y := key[0]+o.dx, key[1]+o.dy
				n2x, n2y := key[2]+o2.dx, key[3]+o2.dy
				if n1x < 0 || n1x >= gmsGrid || n1y < 0 || n1y >= gmsGrid {
					continue
				}
				if n2x < 0 || n2x >= gx2 || n2y < 0 || n2y >= gy2 {
					continue
				}
				S += counts[[4]int{n1x, n1y, n2x, n2y}]
			}
			if S > bestS {
				bestS = S
			}
		}
		// Statistical threshold: expected inlier support grows with sqrt of the
		// mean matches per cell in the support region.
		n := float64(bestS) / 9.0
		thresh := factor * math.Sqrt(n)
		if float64(bestS) > thresh && bestS >= 3 {
			for _, i := range memb {
				accepted[i] = true
			}
		}
	}
}
