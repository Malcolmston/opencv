package objdetect

import (
	"encoding/xml"
	"fmt"
	"os"
	"strings"

	cv "github.com/malcolmston/opencv"
)

// ComputeLBP returns the Local Binary Pattern code image of img (reduced to
// luma). For every pixel the eight 3×3 neighbours are compared with the centre
// and a bit is set for each neighbour that is greater than or equal to it,
// forming a byte in the fixed clockwise order top-left (bit 7), top, top-right,
// right, bottom-right, bottom, bottom-left, left (bit 0). Border pixels use edge
// replication. The result is a single-channel [cv.Mat] the same size as img,
// with each sample the pixel's LBP code (0–255).
//
// LBP codes are the texture primitive behind LBP cascade face detectors and
// LBP-based face recognisers; a histogram of these codes over a region is a
// compact, illumination-robust texture descriptor.
func ComputeLBP(img *cv.Mat) *cv.Mat {
	g := matToGray(img)
	out := cv.NewMat(g.h, g.w, 1)
	// Neighbour offsets and their bit weights, clockwise from top-left.
	type nb struct {
		dx, dy int
		bit    uint8
	}
	nbs := []nb{
		{-1, -1, 128}, {0, -1, 64}, {1, -1, 32}, {1, 0, 16},
		{1, 1, 8}, {0, 1, 4}, {-1, 1, 2}, {-1, 0, 1},
	}
	for y := 0; y < g.h; y++ {
		for x := 0; x < g.w; x++ {
			c := g.at(x, y)
			var code uint8
			for _, n := range nbs {
				if g.at(x+n.dx, y+n.dy) >= c {
					code |= n.bit
				}
			}
			out.Set(y, x, 0, code)
		}
	}
	return out
}

// --- LBP cascade -------------------------------------------------------------

type lbpXMLStorage struct {
	XMLName xml.Name      `xml:"opencv_storage"`
	Cascade lbpXMLCascade `xml:"cascade"`
}

type lbpXMLCascade struct {
	FeatureType string `xml:"featureType"`
	Height      int    `xml:"height"`
	Width       int    `xml:"width"`
	Stages      struct {
		Items []lbpXMLStage `xml:"_"`
	} `xml:"stages"`
	Features struct {
		Items []lbpXMLFeature `xml:"_"`
	} `xml:"features"`
}

type lbpXMLStage struct {
	StageThreshold  float64 `xml:"stageThreshold"`
	WeakClassifiers struct {
		Items []lbpXMLWeak `xml:"_"`
	} `xml:"weakClassifiers"`
}

type lbpXMLWeak struct {
	InternalNodes string `xml:"internalNodes"`
	LeafValues    string `xml:"leafValues"`
}

type lbpXMLFeature struct {
	Rect string `xml:"rect"`
}

type lbpFeatureRT struct {
	x, y, w, h int
}

type lbpStumpRT struct {
	feature     int
	subset      [8]uint32
	left, right float64
}

type lbpStageRT struct {
	threshold float64
	weaks     []lbpStumpRT
}

// LBPCascadeClassifier is a Local Binary Pattern cascade loaded from an OpenCV
// XML file (featureType LBP), the LBP analogue of the Haar [CascadeClassifier].
// Each weak classifier reads a 3×3 grid of cell sums into an 8-bit LBP code and
// looks that code up in a 256-entry subset bitmask to pick a leaf value; a stage
// passes when its accumulated leaf values reach the stage threshold. LBP
// features need no variance normalisation, which makes them cheaper than Haar
// features.
//
// # Deferred / caveats
//
//   - Only the modern <opencv_storage><cascade> layout with featureType LBP and
//     depth-1 (stump) weak classifiers is parsed.
//   - The cell-sum grid uses the loaded rect geometry scaled by the pyramid
//     factor; sub-pixel cell placement is rounded to whole pixels.
type LBPCascadeClassifier struct {
	// ScaleFactor is the ratio between successive window sizes in the scan
	// pyramid. Values <= 1 default to 1.1.
	ScaleFactor float64
	// MinNeighbors, when positive, groups overlapping detections and keeps only
	// clusters with at least this many members.
	MinNeighbors int

	windowW, windowH int
	features         []lbpFeatureRT
	stages           []lbpStageRT
	loaded           bool
}

// Load reads and parses an OpenCV LBP cascade XML file from path.
func (c *LBPCascadeClassifier) Load(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return c.LoadFromString(string(data))
}

// LoadFromString parses an OpenCV LBP cascade from an in-memory XML string. It
// returns an error if the featureType is not LBP or the tree is malformed.
func (c *LBPCascadeClassifier) LoadFromString(xmlData string) error {
	var st lbpXMLStorage
	if err := xml.Unmarshal([]byte(xmlData), &st); err != nil {
		return fmt.Errorf("objdetect: parsing LBP cascade: %w", err)
	}
	cas := st.Cascade
	if ft := strings.ToUpper(strings.TrimSpace(cas.FeatureType)); ft != "" && ft != "LBP" {
		return fmt.Errorf("objdetect: LBPCascadeClassifier requires featureType LBP, got %q", cas.FeatureType)
	}
	if cas.Width <= 0 || cas.Height <= 0 {
		return fmt.Errorf("objdetect: LBP cascade missing window size")
	}
	if len(cas.Stages.Items) == 0 {
		return fmt.Errorf("objdetect: LBP cascade has no stages")
	}
	if len(cas.Features.Items) == 0 {
		return fmt.Errorf("objdetect: LBP cascade has no features")
	}

	features := make([]lbpFeatureRT, 0, len(cas.Features.Items))
	for fi, xf := range cas.Features.Items {
		nums, err := parseFloats(xf.Rect)
		if err != nil || len(nums) < 4 {
			return fmt.Errorf("objdetect: LBP feature %d has malformed rect %q", fi, strings.TrimSpace(xf.Rect))
		}
		features = append(features, lbpFeatureRT{
			x: int(nums[0]), y: int(nums[1]), w: int(nums[2]), h: int(nums[3]),
		})
	}

	stages := make([]lbpStageRT, 0, len(cas.Stages.Items))
	for si, xs := range cas.Stages.Items {
		weaks := make([]lbpStumpRT, 0, len(xs.WeakClassifiers.Items))
		for wi, xw := range xs.WeakClassifiers.Items {
			in, err := parseFloats(xw.InternalNodes)
			if err != nil || len(in) < 11 {
				return fmt.Errorf("objdetect: LBP stage %d weak %d malformed internalNodes", si, wi)
			}
			leaf, err := parseFloats(xw.LeafValues)
			if err != nil || len(leaf) < 2 {
				return fmt.Errorf("objdetect: LBP stage %d weak %d malformed leafValues", si, wi)
			}
			fidx := int(in[2])
			if fidx < 0 || fidx >= len(features) {
				return fmt.Errorf("objdetect: LBP stage %d weak %d feature index %d out of range", si, wi, fidx)
			}
			var subset [8]uint32
			for k := 0; k < 8; k++ {
				subset[k] = uint32(int32(in[3+k]))
			}
			weaks = append(weaks, lbpStumpRT{
				feature: fidx,
				subset:  subset,
				left:    leaf[0],
				right:   leaf[1],
			})
		}
		if len(weaks) == 0 {
			return fmt.Errorf("objdetect: LBP stage %d has no weak classifiers", si)
		}
		stages = append(stages, lbpStageRT{threshold: xs.StageThreshold, weaks: weaks})
	}

	c.windowW = cas.Width
	c.windowH = cas.Height
	c.features = features
	c.stages = stages
	c.loaded = true
	return nil
}

// Loaded reports whether a cascade has been successfully loaded.
func (c *LBPCascadeClassifier) Loaded() bool { return c.loaded }

// WindowSize returns the base detection window (width, height) declared by the
// loaded cascade.
func (c *LBPCascadeClassifier) WindowSize() (w, h int) { return c.windowW, c.windowH }

// DetectMultiScale scans img at growing window sizes and returns the rectangles
// where the LBP cascade fires. The window is grown by ScaleFactor between pyramid
// levels (the image is never resampled) and each candidate is evaluated over an
// integral image. When MinNeighbors is positive, overlapping detections are
// grouped. It panics if no cascade has been loaded.
func (c *LBPCascadeClassifier) DetectMultiScale(img *cv.Mat) []cv.Rect {
	if !c.loaded {
		panic("objdetect: DetectMultiScale on unloaded LBPCascadeClassifier")
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
		return GroupRectangles(raw, c.MinNeighbors, 0.2)
	}
	return raw
}

// evalWindow evaluates every stage of the LBP cascade at window (wx,wy) scaled by
// s and reports whether all stages pass.
func (c *LBPCascadeClassifier) evalWindow(ii *IntegralImage, wx, wy int, s float64) bool {
	for si := range c.stages {
		st := &c.stages[si]
		var stageSum float64
		for wi := range st.weaks {
			stump := &st.weaks[wi]
			code := c.lbpCode(ii, &c.features[stump.feature], wx, wy, s)
			if stump.subset[code>>5]&(1<<(code&31)) != 0 {
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

// lbpCode computes the 8-bit LBP code of the 3×3 cell grid of feature f at window
// (wx,wy) scaled by s, comparing each of the eight surrounding cell sums to the
// centre cell in OpenCV's LBPEvaluator bit order.
func (c *LBPCascadeClassifier) lbpCode(ii *IntegralImage, f *lbpFeatureRT, wx, wy int, s float64) uint32 {
	cw := int(float64(f.w)*s + 0.5)
	ch := int(float64(f.h)*s + 0.5)
	if cw < 1 {
		cw = 1
	}
	if ch < 1 {
		ch = 1
	}
	ox := wx + int(float64(f.x)*s+0.5)
	oy := wy + int(float64(f.y)*s+0.5)
	// 3×3 grid of cell sums; cell(r,c) top-left at (ox+c*cw, oy+r*ch).
	var cell [3][3]float64
	for r := 0; r < 3; r++ {
		for col := 0; col < 3; col++ {
			cell[r][col] = ii.Sum(ox+col*cw, oy+r*ch, cw, ch)
		}
	}
	center := cell[1][1]
	var code uint32
	if cell[0][0] >= center {
		code |= 128
	}
	if cell[0][1] >= center {
		code |= 64
	}
	if cell[0][2] >= center {
		code |= 32
	}
	if cell[1][2] >= center {
		code |= 16
	}
	if cell[2][2] >= center {
		code |= 8
	}
	if cell[2][1] >= center {
		code |= 4
	}
	if cell[2][0] >= center {
		code |= 2
	}
	if cell[1][0] >= center {
		code |= 1
	}
	return code
}
