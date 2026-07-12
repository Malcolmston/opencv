package objdetect

import (
	"encoding/xml"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"

	cv "github.com/malcolmston/opencv"
)

// --- XML binding for the <opencv_storage><cascade> layout ---------------------

type xmlStorage struct {
	XMLName xml.Name   `xml:"opencv_storage"`
	Cascade xmlCascade `xml:"cascade"`
}

type xmlCascade struct {
	StageType   string      `xml:"stageType"`
	FeatureType string      `xml:"featureType"`
	Height      int         `xml:"height"`
	Width       int         `xml:"width"`
	Stages      xmlStages   `xml:"stages"`
	Features    xmlFeatures `xml:"features"`
}

type xmlStages struct {
	Items []xmlStage `xml:"_"`
}

type xmlStage struct {
	MaxWeakCount    int            `xml:"maxWeakCount"`
	StageThreshold  float64        `xml:"stageThreshold"`
	WeakClassifiers xmlWeakClassif `xml:"weakClassifiers"`
}

type xmlWeakClassif struct {
	Items []xmlWeak `xml:"_"`
}

type xmlWeak struct {
	InternalNodes string `xml:"internalNodes"`
	LeafValues    string `xml:"leafValues"`
}

type xmlFeatures struct {
	Items []xmlFeature `xml:"_"`
}

type xmlFeature struct {
	Rects  xmlRects `xml:"rects"`
	Tilted int      `xml:"tilted"`
}

type xmlRects struct {
	Items []string `xml:"_"`
}

// --- runtime model ------------------------------------------------------------

type haarRect struct {
	x, y, w, h int
	weight     float64
}

type haarFeature struct {
	rects  []haarRect
	tilted bool
}

// haarStump is a depth-1 decision tree: compare the feature value against
// threshold·normFactor and emit left or right.
type haarStump struct {
	featureIdx  int
	threshold   float64
	left, right float64
}

type haarStage struct {
	threshold float64
	weaks     []haarStump
}

// CascadeClassifier is a Viola–Jones Haar-feature cascade loaded from an OpenCV
// XML file. Evaluate it over an image with [CascadeClassifier.DetectMultiScale].
//
// # Deferred / caveats
//
//   - Only the modern <opencv_storage><cascade> layout with featureType HAAR
//     and depth-1 (stump) weak classifiers is parsed. The legacy
//     <haarcascade_*><trees> layout, LBP/HOG feature types and multi-depth
//     trees are not supported.
//   - Tilted (45°) Haar features are parsed but evaluated as their upright
//     bounding rectangle, so cascades relying on tilted features lose accuracy.
//   - The variance-normalisation constant follows the classic OpenCV formula,
//     but third-party cascades' exact threshold scaling is not guaranteed to
//     reproduce OpenCV's detections bit-for-bit.
type CascadeClassifier struct {
	// ScaleFactor is the ratio between successive window sizes in the scan
	// pyramid. Values <= 1 are treated as the default 1.1.
	ScaleFactor float64
	// MinNeighbors is the minimum number of overlapping raw detections a
	// grouped detection must have to be kept. Zero (the default) disables
	// grouping and returns every raw hit.
	MinNeighbors int

	windowW, windowH int
	features         []haarFeature
	stages           []haarStage
	loaded           bool
}

// Load reads and parses an OpenCV Haar cascade XML file from path.
func (c *CascadeClassifier) Load(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return c.LoadFromString(string(data))
}

// LoadFromString parses an OpenCV Haar cascade from an in-memory XML string.
func (c *CascadeClassifier) LoadFromString(xmlData string) error {
	var st xmlStorage
	if err := xml.Unmarshal([]byte(xmlData), &st); err != nil {
		return fmt.Errorf("objdetect: parsing cascade: %w", err)
	}
	cas := st.Cascade
	if cas.Width <= 0 || cas.Height <= 0 {
		return fmt.Errorf("objdetect: cascade missing window size")
	}
	if len(cas.Stages.Items) == 0 {
		return fmt.Errorf("objdetect: cascade has no stages")
	}
	if len(cas.Features.Items) == 0 {
		return fmt.Errorf("objdetect: cascade has no features")
	}

	features := make([]haarFeature, 0, len(cas.Features.Items))
	for fi, xf := range cas.Features.Items {
		rects := make([]haarRect, 0, len(xf.Rects.Items))
		for _, rs := range xf.Rects.Items {
			nums, err := parseFloats(rs)
			if err != nil || len(nums) < 5 {
				return fmt.Errorf("objdetect: feature %d has malformed rect %q", fi, strings.TrimSpace(rs))
			}
			rects = append(rects, haarRect{
				x: int(nums[0]), y: int(nums[1]), w: int(nums[2]), h: int(nums[3]),
				weight: nums[4],
			})
		}
		if len(rects) == 0 {
			return fmt.Errorf("objdetect: feature %d has no rects", fi)
		}
		features = append(features, haarFeature{rects: rects, tilted: xf.Tilted != 0})
	}

	stages := make([]haarStage, 0, len(cas.Stages.Items))
	for si, xs := range cas.Stages.Items {
		weaks := make([]haarStump, 0, len(xs.WeakClassifiers.Items))
		for wi, xw := range xs.WeakClassifiers.Items {
			in, err := parseFloats(xw.InternalNodes)
			if err != nil || len(in) < 4 {
				return fmt.Errorf("objdetect: stage %d weak %d malformed internalNodes", si, wi)
			}
			leaf, err := parseFloats(xw.LeafValues)
			if err != nil || len(leaf) < 2 {
				return fmt.Errorf("objdetect: stage %d weak %d malformed leafValues", si, wi)
			}
			fidx := int(in[2])
			if fidx < 0 || fidx >= len(features) {
				return fmt.Errorf("objdetect: stage %d weak %d feature index %d out of range", si, wi, fidx)
			}
			weaks = append(weaks, haarStump{
				featureIdx: fidx,
				threshold:  in[3],
				left:       leaf[0],
				right:      leaf[1],
			})
		}
		if len(weaks) == 0 {
			return fmt.Errorf("objdetect: stage %d has no weak classifiers", si)
		}
		stages = append(stages, haarStage{threshold: xs.StageThreshold, weaks: weaks})
	}

	c.windowW = cas.Width
	c.windowH = cas.Height
	c.features = features
	c.stages = stages
	c.loaded = true
	return nil
}

// Loaded reports whether a cascade has been successfully loaded.
func (c *CascadeClassifier) Loaded() bool { return c.loaded }

// WindowSize returns the base detection window (width, height) declared by the
// loaded cascade.
func (c *CascadeClassifier) WindowSize() (w, h int) { return c.windowW, c.windowH }

// DetectMultiScale scans img at increasing window sizes and returns the
// rectangles where the cascade fires. The classifier window is grown by
// ScaleFactor between levels (the image itself is never resampled), and each
// window is evaluated with an integral image for constant-time feature sums.
// When MinNeighbors is positive, overlapping raw detections are grouped and
// only clusters with at least MinNeighbors members survive. It panics if no
// cascade has been loaded.
func (c *CascadeClassifier) DetectMultiScale(img *cv.Mat) []cv.Rect {
	if !c.loaded {
		panic("objdetect: DetectMultiScale on unloaded CascadeClassifier")
	}
	ii := NewIntegralImage(img)
	sf := c.ScaleFactor
	if sf <= 1 {
		sf = 1.1
	}

	var raw []cv.Rect
	s := 1.0
	for {
		sw := int(float64(c.windowW)*s + 0.5)
		sh := int(float64(c.windowH)*s + 0.5)
		if sw > ii.W || sh > ii.H {
			break
		}
		step := int(s + 0.5)
		if step < 1 {
			step = 1
		}
		for y := 0; y+sh <= ii.H; y += step {
			for x := 0; x+sw <= ii.W; x += step {
				if c.evalWindow(ii, x, y, s) {
					raw = append(raw, cv.Rect{X: x, Y: y, Width: sw, Height: sh})
				}
			}
		}
		s *= sf
	}

	if c.MinNeighbors > 0 {
		return groupRectangles(raw, c.MinNeighbors)
	}
	return raw
}

// evalWindow runs the full cascade at window (wx,wy) scaled by s and reports
// whether every stage passes.
func (c *CascadeClassifier) evalWindow(ii *IntegralImage, wx, wy int, s float64) bool {
	sw := int(float64(c.windowW)*s + 0.5)
	sh := int(float64(c.windowH)*s + 0.5)
	area := float64(sw * sh)
	sum := ii.Sum(wx, wy, sw, sh)
	sq := ii.SqSum(wx, wy, sw, sh)

	// normFactor = stddev * area = sqrt(area*Σx² − (Σx)²).
	nf := area*sq - sum*sum
	if nf > 0 {
		nf = math.Sqrt(nf)
	} else {
		nf = 1
	}

	for si := range c.stages {
		st := &c.stages[si]
		var stageSum float64
		for wi := range st.weaks {
			stump := &st.weaks[wi]
			f := &c.features[stump.featureIdx]
			var fsum float64
			for _, r := range f.rects {
				rx := wx + int(float64(r.x)*s+0.5)
				ry := wy + int(float64(r.y)*s+0.5)
				rw := int(float64(r.w)*s + 0.5)
				rh := int(float64(r.h)*s + 0.5)
				if rw <= 0 || rh <= 0 {
					continue
				}
				fsum += r.weight * ii.Sum(rx, ry, rw, rh)
			}
			if fsum < stump.threshold*nf {
				stageSum += stump.left
			} else {
				stageSum += stump.right
			}
		}
		if stageSum < st.threshold {
			return false
		}
	}
	return true
}

// parseFloats splits s on whitespace and parses each field as a float64. The
// OpenCV writer emits values such as "0." and "-1.2345e-01" which strconv
// accepts directly.
func parseFloats(s string) ([]float64, error) {
	fields := strings.Fields(s)
	out := make([]float64, 0, len(fields))
	for _, f := range fields {
		v, err := strconv.ParseFloat(f, 64)
		if err != nil {
			return nil, err
		}
		out = append(out, v)
	}
	return out, nil
}

// groupRectangles clusters similar rectangles (as OpenCV's groupRectangles
// does) and returns one averaged rectangle per cluster that has at least
// minNeighbors members.
func groupRectangles(rects []cv.Rect, minNeighbors int) []cv.Rect {
	n := len(rects)
	if n == 0 {
		return nil
	}
	const eps = 0.2
	label := make([]int, n)
	for i := range label {
		label[i] = -1
	}
	similar := func(a, b cv.Rect) bool {
		delta := eps * float64(minInt(a.Width, b.Width)+minInt(a.Height, b.Height)) * 0.5
		return math.Abs(float64(a.X-b.X)) <= delta &&
			math.Abs(float64(a.Y-b.Y)) <= delta &&
			math.Abs(float64(a.Width-b.Width)) <= delta &&
			math.Abs(float64(a.Height-b.Height)) <= delta
	}
	nClusters := 0
	for i := 0; i < n; i++ {
		if label[i] != -1 {
			continue
		}
		label[i] = nClusters
		for j := i + 1; j < n; j++ {
			if label[j] == -1 && similar(rects[i], rects[j]) {
				label[j] = nClusters
			}
		}
		nClusters++
	}

	type acc struct {
		x, y, w, h, count int
	}
	accs := make([]acc, nClusters)
	for i, l := range label {
		accs[l].x += rects[i].X
		accs[l].y += rects[i].Y
		accs[l].w += rects[i].Width
		accs[l].h += rects[i].Height
		accs[l].count++
	}

	var out []cv.Rect
	for _, a := range accs {
		if a.count < minNeighbors {
			continue
		}
		out = append(out, cv.Rect{
			X:      a.x / a.count,
			Y:      a.y / a.count,
			Width:  a.w / a.count,
			Height: a.h / a.count,
		})
	}
	return out
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}
