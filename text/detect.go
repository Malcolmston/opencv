package text

import (
	cv "github.com/malcolmston/opencv"
)

// TextDetectorParams bundles the parameters of the full classical detection
// pipeline driven by [DetectRegions] and [DetectTextLines]: MSER region proposal,
// the two-stage Extremal Region classifier, and geometric grouping.
type TextDetectorParams struct {
	// MSER controls region proposal (see [MSERParams]).
	MSER MSERParams
	// NM1 and NM2 are the two Extremal Region classifier stages applied in order.
	NM1 ERFilterNM1
	NM2 ERFilterNM2
	// Orientation selects the grouping mode (see [GroupingOrientation]).
	Orientation GroupingOrientation
	// MinLineSize drops grouped lines with fewer than this many characters. 1
	// keeps isolated characters; 2 requires at least a pair.
	MinLineSize int
}

// DefaultTextDetectorParams returns a pipeline configuration tuned for dark
// characters on a lighter background (and, via MSER's dual polarity, the
// reverse), horizontal reading order, and lines of at least two characters.
func DefaultTextDetectorParams() TextDetectorParams {
	return TextDetectorParams{
		MSER:        DefaultMSERParams(),
		NM1:         DefaultERFilterNM1(),
		NM2:         DefaultERFilterNM2(),
		Orientation: OrientationHoriz,
		MinLineSize: 1,
	}
}

// DetectTextLines runs the complete pipeline and returns the surviving character
// boxes grouped into text lines: extremal-region proposal ([MSERRegionsWithParams]),
// two-stage filtering ([ERFilterNM1] then [ERFilterNM2]) and grouping
// ([ERGrouping]). Each line is sorted left-to-right and lines run top-to-bottom.
func DetectTextLines(img *cv.Mat, p TextDetectorParams) [][]cv.Rect {
	regions := MSERRegionsWithParams(img, p.MSER)
	regions = p.NM1.Filter(regions)
	regions = p.NM2.Filter(regions)

	boxes := make([]cv.Rect, len(regions))
	for i, r := range regions {
		boxes[i] = r.Rect
	}
	lines := ERGrouping(boxes, p.Orientation)

	if p.MinLineSize <= 1 {
		return lines
	}
	var kept [][]cv.Rect
	for _, l := range lines {
		if len(l) >= p.MinLineSize {
			kept = append(kept, l)
		}
	}
	return kept
}

// DetectRegions is the convenience entry point of the module: it runs the full
// pipeline (see [DetectTextLines]) and returns one bounding box per detected text
// line, ordered top-to-bottom. It is the closest analogue to OpenCV's
// cv::text::detectRegions/erGrouping one-call helper.
func DetectRegions(img *cv.Mat, p TextDetectorParams) []cv.Rect {
	lines := DetectTextLines(img, p)
	out := make([]cv.Rect, 0, len(lines))
	for _, l := range lines {
		out = append(out, LineBoundingBox(l))
	}
	return out
}
