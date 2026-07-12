package features2d

import (
	"math"
	"sort"

	cv "github.com/malcolmston/opencv"
)

// Default ORB parameters.
const (
	defaultNFeatures     = 500
	defaultFastThreshold = 20
	defaultHarrisK       = 0.04
	harrisBlockSize      = 7
)

// ORB is an Oriented FAST and Rotated BRIEF detector and descriptor. It detects
// FAST corners (via cv.FASTCorners), ranks them by a Harris corner response,
// assigns each keypoint an orientation from its patch intensity centroid, and
// computes a steered BRIEF descriptor. The resulting binary descriptors are
// invariant to translation and in-plane rotation.
//
// The zero value is usable and applies the defaults; construct a customised
// instance with [NewORB].
type ORB struct {
	// NFeatures is the maximum number of keypoints to retain, keeping the
	// strongest by Harris response. Zero or negative means the default (500);
	// use a very large value to keep all corners.
	NFeatures int
	// FastThreshold is the intensity threshold passed to cv.FASTCorners. Zero
	// means the default (20).
	FastThreshold int
	// PatchSize is the descriptor patch side length in pixels. Zero means the
	// default (31). It also sets the orientation and BRIEF sampling radius.
	PatchSize int
	// brief carries the steered BRIEF pattern used for description.
	brief *BRIEF
}

// NewORB returns an ORB detector retaining up to nFeatures keypoints with the
// default FAST threshold and patch size. Pass nFeatures <= 0 to keep all
// detected corners.
func NewORB(nFeatures int) *ORB {
	return &ORB{NFeatures: nFeatures}
}

func (o *ORB) nFeatures() int {
	if o.NFeatures > 0 {
		return o.NFeatures
	}
	return defaultNFeatures
}

func (o *ORB) fastThreshold() int {
	if o.FastThreshold > 0 {
		return o.FastThreshold
	}
	return defaultFastThreshold
}

func (o *ORB) patchSize() int {
	if o.PatchSize > 0 {
		return o.PatchSize
	}
	return defaultPatchSize
}

func (o *ORB) briefExtractor() *BRIEF {
	if o.brief == nil {
		return &BRIEF{}
	}
	return o.brief
}

// Detect finds oriented FAST keypoints in img without computing descriptors.
// The image may be single- or three-channel. Keypoints closer to the border
// than the patch radius are discarded so their orientation patch fits, the
// survivors are ranked by Harris response, and at most NFeatures are returned in
// descending response order.
func (o *ORB) Detect(img *cv.Mat) []KeyPoint {
	gray := toGray(img)
	half := o.patchSize() / 2
	corners := cv.FASTCorners(gray, o.fastThreshold(), true)
	harris := cv.CornerHarris(gray, harrisBlockSize, 3, defaultHarrisK)

	kps := make([]KeyPoint, 0, len(corners))
	for _, p := range corners {
		if p.X < half || p.Y < half || p.X >= gray.Cols-half || p.Y >= gray.Rows-half {
			continue
		}
		kps = append(kps, KeyPoint{
			Pt:       p,
			Size:     float64(o.patchSize()),
			Angle:    orientation(gray, p.X, p.Y, half),
			Response: harris.At(p.Y, p.X),
			Octave:   0,
		})
	}
	// Rank by Harris response, strongest first; ties broken deterministically by
	// position so the order never depends on detection order.
	sort.SliceStable(kps, func(i, j int) bool {
		if kps[i].Response != kps[j].Response {
			return kps[i].Response > kps[j].Response
		}
		if kps[i].Pt.Y != kps[j].Pt.Y {
			return kps[i].Pt.Y < kps[j].Pt.Y
		}
		return kps[i].Pt.X < kps[j].Pt.X
	})
	if n := o.nFeatures(); len(kps) > n {
		kps = kps[:n]
	}
	return kps
}

// DetectAndCompute detects oriented FAST keypoints in img and computes their
// rotated BRIEF descriptors, returning parallel slices: the i-th descriptor row
// describes the i-th keypoint. The image may be single- or three-channel.
func (o *ORB) DetectAndCompute(img *cv.Mat) ([]KeyPoint, [][]byte) {
	kps := o.Detect(img)
	if len(kps) == 0 {
		return nil, nil
	}
	_, desc := o.briefExtractor().Compute(img, kps)
	return kps, desc
}

// orientation computes the keypoint angle in degrees [0,360) from the intensity
// centroid of the circular patch of the given radius centred on (cx, cy):
// angle = atan2(m01, m10) where m10 and m01 are the first-order image moments.
func orientation(gray *cv.Mat, cx, cy, radius int) float64 {
	var m10, m01 float64
	r2 := radius * radius
	for dy := -radius; dy <= radius; dy++ {
		y := cy + dy
		if y < 0 || y >= gray.Rows {
			continue
		}
		for dx := -radius; dx <= radius; dx++ {
			if dx*dx+dy*dy > r2 {
				continue
			}
			x := cx + dx
			if x < 0 || x >= gray.Cols {
				continue
			}
			v := float64(gray.Data[y*gray.Cols+x])
			m10 += float64(dx) * v
			m01 += float64(dy) * v
		}
	}
	deg := math.Atan2(m01, m10) * 180 / math.Pi
	if deg < 0 {
		deg += 360
	}
	return deg
}
