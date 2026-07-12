package linedescriptor

import "sort"

// FilterByLength returns the subset of lines whose [KeyLine.Length] is at least
// minLength, preserving order. It is a convenience for discarding short,
// noise-level segments after detection. The input slice is not modified.
func FilterByLength(lines []KeyLine, minLength float64) []KeyLine {
	out := make([]KeyLine, 0, len(lines))
	for _, kl := range lines {
		if kl.Length >= minLength {
			out = append(out, kl)
		}
	}
	return out
}

// FilterByResponse returns the subset of lines whose [KeyLine.Response] is at
// least minResponse, preserving order. The detector sets Response to segment
// length, so this selects the most prominent segments. The input is not modified.
func FilterByResponse(lines []KeyLine, minResponse float64) []KeyLine {
	out := make([]KeyLine, 0, len(lines))
	for _, kl := range lines {
		if kl.Response >= minResponse {
			out = append(out, kl)
		}
	}
	return out
}

// TopN returns at most the n highest-[KeyLine.Response] segments from lines,
// sorted by descending response (ties broken by longer length, then by
// smaller start-point coordinates) for a deterministic result. If n is zero or
// negative the result is empty; if n exceeds the input size every segment is
// returned. The input slice is not modified.
func TopN(lines []KeyLine, n int) []KeyLine {
	if n <= 0 {
		return nil
	}
	sorted := make([]KeyLine, len(lines))
	copy(sorted, lines)
	sort.SliceStable(sorted, func(i, j int) bool {
		if sorted[i].Response != sorted[j].Response {
			return sorted[i].Response > sorted[j].Response
		}
		if sorted[i].Length != sorted[j].Length {
			return sorted[i].Length > sorted[j].Length
		}
		if sorted[i].StartPoint.Y != sorted[j].StartPoint.Y {
			return sorted[i].StartPoint.Y < sorted[j].StartPoint.Y
		}
		return sorted[i].StartPoint.X < sorted[j].StartPoint.X
	})
	if n > len(sorted) {
		n = len(sorted)
	}
	return sorted[:n]
}

// FilterByOctave returns the subset of multi-octave segments detected in the
// given octave, preserving order. It complements [LSDDetector.DetectPyramid] by
// letting callers select a single scale level. The input is not modified.
func FilterByOctave(lines []KeyLineEx, octave int) []KeyLineEx {
	out := make([]KeyLineEx, 0, len(lines))
	for _, l := range lines {
		if l.Octave == octave {
			out = append(out, l)
		}
	}
	return out
}

// FilterCodesByLength keeps the (line, code) pairs whose line is at least
// minLength long, returning the filtered lines and their aligned codes. It
// panics if lines and codes have different lengths. This lets a caller prune a
// descriptor set and its keylines together after [BinaryDescriptor.Compute].
func FilterCodesByLength(lines []KeyLine, codes [][]byte, minLength float64) ([]KeyLine, [][]byte) {
	if len(lines) != len(codes) {
		panic("linedescriptor: FilterCodesByLength lines and codes length mismatch")
	}
	outLines := make([]KeyLine, 0, len(lines))
	outCodes := make([][]byte, 0, len(codes))
	for i, kl := range lines {
		if kl.Length >= minLength {
			outLines = append(outLines, kl)
			outCodes = append(outCodes, codes[i])
		}
	}
	return outLines, outCodes
}
