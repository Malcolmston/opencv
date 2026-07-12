package hfs

import (
	"testing"

	cv "github.com/malcolmston/opencv"
)

// quadrantImage builds a size x size RGB image split into four solid-colour
// quadrants (red, green, blue, yellow). It is the canonical multi-region test
// input: the four colours are far apart in L*a*b* space, so a correct pipeline
// must recover exactly four regions.
func quadrantImage(size int) *cv.Mat {
	m := cv.NewMat(size, size, 3)
	half := size / 2
	colors := [4][3]uint8{
		{220, 30, 30},  // top-left  red
		{30, 200, 30},  // top-right green
		{30, 30, 220},  // bot-left  blue
		{230, 220, 40}, // bot-right yellow
	}
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			q := 0
			if x >= half {
				q++
			}
			if y >= half {
				q += 2
			}
			m.SetPixel(y, x, colors[q][:])
		}
	}
	return m
}

// componentCount counts the 4-connected components of a per-pixel labelling. It
// is used to verify that every emitted label is a single connected region: when
// every label owns exactly one component the total component count equals the
// number of distinct labels.
func componentCount(labels []int, rows, cols int) (components int, distinctLabels int) {
	seen := make(map[int]struct{})
	visited := make([]bool, len(labels))
	stack := make([]int, 0, 64)
	for start := 0; start < len(labels); start++ {
		seen[labels[start]] = struct{}{}
		if visited[start] {
			continue
		}
		components++
		lbl := labels[start]
		visited[start] = true
		stack = stack[:0]
		stack = append(stack, start)
		for len(stack) > 0 {
			idx := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			y, x := idx/cols, idx%cols
			for _, o := range neighbors4 {
				nx, ny := x+o.dx, y+o.dy
				if nx < 0 || nx >= cols || ny < 0 || ny >= rows {
					continue
				}
				nidx := ny*cols + nx
				if !visited[nidx] && labels[nidx] == lbl {
					visited[nidx] = true
					stack = append(stack, nidx)
				}
			}
		}
	}
	return components, len(seen)
}

func TestSegmentFourQuadrants(t *testing.T) {
	img := quadrantImage(64)
	seg := CreateWithDefaults(img.Rows, img.Cols)
	seg.SetSlicSpixelSize(8)
	// The default absorption sizes (100/200) would swallow whole quadrants of a
	// 64x64 image (each is 1024 px, but stage-II absorbs anything below 200 px
	// only — quadrants survive). Keep defaults to exercise the real thresholds.
	drawn := seg.PerformSegmentCpu(img, true)
	if drawn.Channels != 3 {
		t.Fatalf("drawn image channels = %d, want 3", drawn.Channels)
	}
	if got := seg.NumSegments(); got != 4 {
		t.Fatalf("NumSegments = %d, want 4", got)
	}
}

func TestEveryLabelIsConnected(t *testing.T) {
	img := quadrantImage(64)
	seg := CreateWithDefaults(img.Rows, img.Cols)
	seg.PerformSegmentCpu(img, false)
	labels, rows, cols := seg.Labels()
	comps, distinct := componentCount(labels, rows, cols)
	if comps != distinct {
		t.Fatalf("labels form %d connected components for %d distinct labels; each label must be one component", comps, distinct)
	}
	if distinct != seg.NumSegments() {
		t.Fatalf("distinct labels = %d, NumSegments = %d", distinct, seg.NumSegments())
	}
}

func TestDeterministic(t *testing.T) {
	img := quadrantImage(48)

	run := func() []int {
		seg := CreateWithDefaults(img.Rows, img.Cols)
		seg.PerformSegmentCpu(img, false)
		l, _, _ := seg.Labels()
		return l
	}
	a := run()
	b := run()
	if len(a) != len(b) {
		t.Fatalf("label length mismatch %d vs %d", len(a), len(b))
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("non-deterministic labelling at pixel %d: %d vs %d", i, a[i], b[i])
		}
	}
}

func TestDrawModesDeterministic(t *testing.T) {
	img := quadrantImage(48)
	seg := CreateWithDefaults(img.Rows, img.Cols)
	seg.PerformSegmentCpu(img, true)

	for _, mode := range []DrawMode{DrawAverageColor, DrawRandomColor} {
		a := seg.DrawSegmentation(mode)
		b := seg.DrawSegmentation(mode)
		if len(a.Data) != len(b.Data) {
			t.Fatalf("mode %d: data length mismatch", mode)
		}
		for i := range a.Data {
			if a.Data[i] != b.Data[i] {
				t.Fatalf("mode %d: non-deterministic rendering at %d", mode, i)
			}
		}
	}
}

func TestAverageColorMatchesRegions(t *testing.T) {
	img := quadrantImage(64)
	seg := CreateWithDefaults(img.Rows, img.Cols)
	seg.PerformSegmentCpu(img, true)
	avg := seg.DrawSegmentation(DrawAverageColor)

	// Each quadrant is solid, so the average-colour rendering must reproduce the
	// original colour at the centre of every quadrant almost exactly.
	centers := [][2]int{{16, 16}, {16, 48}, {48, 16}, {48, 48}}
	for _, c := range centers {
		y, x := c[0], c[1]
		for k := 0; k < 3; k++ {
			got := int(avg.At(y, x, k))
			want := int(img.At(y, x, k))
			if diff := got - want; diff < -4 || diff > 4 {
				t.Fatalf("avg at (%d,%d) ch %d = %d, want ~%d", y, x, k, got, want)
			}
		}
	}
}

func TestGrayInputPromoted(t *testing.T) {
	// A grayscale image with two horizontal bands must yield two regions.
	m := cv.NewMat(40, 40, 1)
	for y := 0; y < 40; y++ {
		v := uint8(40)
		if y >= 20 {
			v = 210
		}
		for x := 0; x < 40; x++ {
			m.Set(y, x, 0, v)
		}
	}
	seg := CreateWithDefaults(m.Rows, m.Cols)
	seg.SetSlicSpixelSize(6)
	seg.PerformSegmentCpu(m, false)
	if got := seg.NumSegments(); got != 2 {
		t.Fatalf("gray two-band NumSegments = %d, want 2", got)
	}
	labels, rows, cols := seg.Labels()
	comps, distinct := componentCount(labels, rows, cols)
	if comps != distinct {
		t.Fatalf("gray labels not all connected: %d comps, %d labels", comps, distinct)
	}
}

func TestSmallRegionAbsorption(t *testing.T) {
	// A large background with a tiny 3x3 speckle of a different colour. With a
	// generous minRegionSize the speckle must be absorbed into the background,
	// leaving a single region.
	m := cv.NewMat(40, 40, 3)
	for y := 0; y < 40; y++ {
		for x := 0; x < 40; x++ {
			m.SetPixel(y, x, []uint8{120, 120, 120})
		}
	}
	for y := 18; y < 21; y++ {
		for x := 18; x < 21; x++ {
			m.SetPixel(y, x, []uint8{240, 10, 10})
		}
	}
	seg := Create(m.Rows, m.Cols,
		DefaultSegEgbThresholdI, 800,
		DefaultSegEgbThresholdII, 800,
		DefaultSpatialWeight, 6, DefaultNumSlicIter)
	seg.PerformSegmentCpu(m, false)
	if got := seg.NumSegments(); got != 1 {
		t.Fatalf("speckle absorption NumSegments = %d, want 1", got)
	}
}

func TestAccessors(t *testing.T) {
	seg := Create(10, 12, 0.1, 50, 0.3, 60, 0.7, 9, 3)
	if seg.Height() != 10 || seg.Width() != 12 {
		t.Fatalf("size accessors wrong: %dx%d", seg.Height(), seg.Width())
	}
	checks := []struct {
		name string
		set  func()
		get  func() float64
		want float64
	}{
		{"SegEgbThresholdI", func() { seg.SetSegEgbThresholdI(0.11) }, seg.GetSegEgbThresholdI, 0.11},
		{"SegEgbThresholdII", func() { seg.SetSegEgbThresholdII(0.22) }, seg.GetSegEgbThresholdII, 0.22},
		{"SpatialWeight", func() { seg.SetSpatialWeight(0.9) }, seg.GetSpatialWeight, 0.9},
	}
	for _, c := range checks {
		c.set()
		if got := c.get(); got != c.want {
			t.Fatalf("%s = %v, want %v", c.name, got, c.want)
		}
	}
	intChecks := []struct {
		name string
		set  func()
		get  func() int
		want int
	}{
		{"MinRegionSizeI", func() { seg.SetMinRegionSizeI(77) }, seg.GetMinRegionSizeI, 77},
		{"MinRegionSizeII", func() { seg.SetMinRegionSizeII(88) }, seg.GetMinRegionSizeII, 88},
		{"SlicSpixelSize", func() { seg.SetSlicSpixelSize(5) }, seg.GetSlicSpixelSize, 5},
		{"NumSlicIter", func() { seg.SetNumSlicIter(7) }, seg.GetNumSlicIter, 7},
	}
	for _, c := range intChecks {
		c.set()
		if got := c.get(); got != c.want {
			t.Fatalf("%s = %v, want %v", c.name, got, c.want)
		}
	}
}

func TestGpuMatchesCpu(t *testing.T) {
	img := quadrantImage(48)
	c := CreateWithDefaults(img.Rows, img.Cols)
	cpu := c.PerformSegmentCpu(img, true)
	g := CreateWithDefaults(img.Rows, img.Cols)
	gpu := g.PerformSegmentGpu(img, true)
	for i := range cpu.Data {
		if cpu.Data[i] != gpu.Data[i] {
			t.Fatalf("Gpu and Cpu results differ at %d", i)
		}
	}
}

func TestLabelsBeforeSegment(t *testing.T) {
	seg := CreateWithDefaults(8, 8)
	if l, r, c := seg.Labels(); l != nil || r != 0 || c != 0 {
		t.Fatalf("Labels before segment = %v %d %d, want nil 0 0", l, r, c)
	}
	if seg.NumSegments() != 0 {
		t.Fatalf("NumSegments before segment = %d, want 0", seg.NumSegments())
	}
}

func TestPanics(t *testing.T) {
	mustPanic := func(name string, fn func()) {
		defer func() {
			if recover() == nil {
				t.Fatalf("%s did not panic", name)
			}
		}()
		fn()
	}
	mustPanic("Create zero size", func() { Create(0, 5, 0.1, 1, 0.2, 1, 0.5, 4, 1) })
	mustPanic("empty image", func() {
		CreateWithDefaults(4, 4).PerformSegmentCpu(&cv.Mat{}, true)
	})
	mustPanic("size mismatch", func() {
		CreateWithDefaults(4, 4).PerformSegmentCpu(cv.NewMat(5, 5, 3), true)
	})
	mustPanic("draw before segment", func() {
		CreateWithDefaults(4, 4).DrawSegmentation(DrawAverageColor)
	})
}
