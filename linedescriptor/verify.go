package linedescriptor

import "math"

// GeometricMatchParams controls the geometric consistency test applied by
// [MatchLineSegments] on top of the descriptor (appearance) distance. A pair of
// segments is accepted only when it passes every enabled constraint, which
// suppresses the appearance-only false matches that occur between visually
// similar but geometrically incompatible lines.
type GeometricMatchParams struct {
	// MaxDistance is the largest admissible Hamming distance between the two
	// segments' binary codes.
	MaxDistance int
	// MaxAngleDiff is the largest admissible orientation difference in radians,
	// compared modulo π so that opposite directions count as equal. A value <= 0
	// disables the orientation test.
	MaxAngleDiff float64
	// MaxLengthRatio, when greater than 1, requires the longer of the two
	// segments to be at most MaxLengthRatio times the shorter. A value <= 1
	// disables the length test.
	MaxLengthRatio float64
}

// DefaultGeometricMatchParams returns a lenient-but-useful configuration: a
// 12-bit Hamming ceiling, a 15° orientation tolerance and a 2× length-ratio
// ceiling.
func DefaultGeometricMatchParams() GeometricMatchParams {
	return GeometricMatchParams{
		MaxDistance:    12,
		MaxAngleDiff:   15 * math.Pi / 180,
		MaxLengthRatio: 2,
	}
}

// MatchLineSegments matches two sets of line segments using both appearance and
// geometry, mirroring the geometric verification stage that upstream line
// matching applies after the raw descriptor match. For each query segment it
// picks the train segment of minimum Hamming distance among those that also
// satisfy the orientation and length constraints in params; ties break by the
// lower train index. Queries with no geometrically consistent train segment are
// omitted from the result, so the returned matches are a filtered, verified
// subset rather than one entry per query.
//
// lines1/codes1 and lines2/codes2 must be aligned by index (as returned
// together by [BinaryDescriptor.Compute]); the function panics if a lines slice
// and its codes slice differ in length.
func MatchLineSegments(lines1 []KeyLine, codes1 [][]byte, lines2 []KeyLine, codes2 [][]byte, params GeometricMatchParams) []DMatch {
	if len(lines1) != len(codes1) || len(lines2) != len(codes2) {
		panic("linedescriptor: MatchLineSegments lines and codes length mismatch")
	}
	var out []DMatch
	for qi := range lines1 {
		best := -1
		bestDist := math.MaxInt
		for ti := range lines2 {
			if !geometricallyConsistent(lines1[qi], lines2[ti], params) {
				continue
			}
			d := HammingDistance(codes1[qi], codes2[ti])
			if d > params.MaxDistance {
				continue
			}
			if d < bestDist {
				bestDist = d
				best = ti
			}
		}
		if best >= 0 {
			out = append(out, DMatch{QueryIdx: qi, TrainIdx: best, Distance: bestDist})
		}
	}
	return out
}

// geometricallyConsistent reports whether segments a and b satisfy the
// orientation and length constraints in params (the Hamming test is applied
// separately by the caller).
func geometricallyConsistent(a, b KeyLine, params GeometricMatchParams) bool {
	if params.MaxAngleDiff > 0 {
		d := angleDiff(a.Angle, b.Angle)
		if d > math.Pi/2 {
			d = math.Pi - d // treat opposite directions as equal
		}
		if d > params.MaxAngleDiff {
			return false
		}
	}
	if params.MaxLengthRatio > 1 {
		la, lb := a.Length, b.Length
		if la <= 0 || lb <= 0 {
			return false
		}
		hi, lo := la, lb
		if lb > la {
			hi, lo = lb, la
		}
		if hi/lo > params.MaxLengthRatio {
			return false
		}
	}
	return true
}

// SegmentOverlap returns the fraction, in [0, 1], of the shorter segment's
// projected extent that overlaps the longer segment when both are projected
// onto the longer segment's supporting line. It is a geometric similarity cue:
// collinear, overlapping segments score near 1 while distant or crossing
// segments score near 0. Zero-length segments yield 0.
func SegmentOverlap(a, b KeyLine) float64 {
	// Use the longer segment to define the projection axis.
	long, short := a, b
	if b.Length > a.Length {
		long, short = b, a
	}
	if long.Length <= 0 {
		return 0
	}
	ax := float64(long.StartPoint.X)
	ay := float64(long.StartPoint.Y)
	dirX := float64(long.EndPoint.X-long.StartPoint.X) / long.Length
	dirY := float64(long.EndPoint.Y-long.StartPoint.Y) / long.Length

	proj := func(x, y int) float64 {
		return (float64(x)-ax)*dirX + (float64(y)-ay)*dirY
	}
	s0 := proj(short.StartPoint.X, short.StartPoint.Y)
	s1 := proj(short.EndPoint.X, short.EndPoint.Y)
	if s0 > s1 {
		s0, s1 = s1, s0
	}
	l0, l1 := 0.0, long.Length

	overlap := math.Min(s1, l1) - math.Max(s0, l0)
	if overlap <= 0 {
		return 0
	}
	shorterExtent := math.Min(short.Length, long.Length)
	if shorterExtent <= 0 {
		return 0
	}
	return math.Min(1, overlap/shorterExtent)
}
