package cudaimgproc_test

import (
	"testing"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/cudaimgproc"
)

// makeGray builds a single-channel test image with a deterministic, varied
// intensity pattern.
func makeGray(rows, cols int) *cv.Mat {
	m := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			m.Data[y*cols+x] = uint8((x*7 + y*13 + x*y) % 256)
		}
	}
	return m
}

// makeRGB builds a three-channel test image with a deterministic pattern.
func makeRGB(rows, cols int) *cv.Mat {
	m := cv.NewMat(rows, cols, 3)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			i := (y*cols + x) * 3
			m.Data[i+0] = uint8((x * 11) % 256)
			m.Data[i+1] = uint8((y * 17) % 256)
			m.Data[i+2] = uint8((x + y) * 5 % 256)
		}
	}
	return m
}

func upload(t *testing.T, m *cv.Mat) cudaimgproc.GpuMat {
	t.Helper()
	var g cudaimgproc.GpuMat
	g.Upload(m)
	return g
}

func mustEqual(t *testing.T, got, want *cv.Mat) {
	t.Helper()
	if got.Rows != want.Rows || got.Cols != want.Cols || got.Channels != want.Channels {
		t.Fatalf("shape mismatch: got %dx%dx%d want %dx%dx%d",
			got.Rows, got.Cols, got.Channels, want.Rows, want.Cols, want.Channels)
	}
	if len(got.Data) != len(want.Data) {
		t.Fatalf("data length mismatch: %d vs %d", len(got.Data), len(want.Data))
	}
	for i := range got.Data {
		if got.Data[i] != want.Data[i] {
			t.Fatalf("data mismatch at %d: got %d want %d", i, got.Data[i], want.Data[i])
		}
	}
}

func TestGpuMatRoundTrip(t *testing.T) {
	src := makeRGB(8, 6)
	g := upload(t, src)
	if g.Empty() {
		t.Fatal("uploaded GpuMat should not be empty")
	}
	r, c := g.Size()
	if r != 8 || c != 6 {
		t.Fatalf("Size = (%d,%d), want (8,6)", r, c)
	}
	if g.Channels() != 3 {
		t.Fatalf("Channels = %d, want 3", g.Channels())
	}
	got := g.Download()
	mustEqual(t, got, src)

	// Upload stores a clone: mutating the source must not affect the GpuMat.
	src.Data[0] ^= 0xFF
	got2 := g.Download()
	if got2.Data[0] == src.Data[0] {
		t.Fatal("Upload should have cloned the source")
	}
}

func TestGpuMatCloneAndRelease(t *testing.T) {
	g := upload(t, makeGray(4, 4))
	cl := g.Clone()
	g.Release()
	if !g.Empty() {
		t.Fatal("released GpuMat should be empty")
	}
	if cl.Empty() {
		t.Fatal("clone must survive release of the original")
	}
	if cl.Channels() != 1 {
		t.Fatalf("clone Channels = %d, want 1", cl.Channels())
	}
	empty := cudaimgproc.NewGpuMat()
	if !empty.Empty() {
		t.Fatal("NewGpuMat should be empty")
	}
	if empty.Channels() != 0 {
		t.Fatalf("empty Channels = %d, want 0", empty.Channels())
	}
	if got := cl.Clone(); got.Empty() {
		t.Fatal("clone of non-empty should not be empty")
	}
	if got := empty.Clone(); !got.Empty() {
		t.Fatal("clone of empty should be empty")
	}
}

func TestNewGpuMatWithSize(t *testing.T) {
	g := cudaimgproc.NewGpuMatWithSize(3, 5, 3)
	r, c := g.Size()
	if r != 3 || c != 5 || g.Channels() != 3 {
		t.Fatalf("unexpected size %dx%dx%d", r, c, g.Channels())
	}
}

func TestCvtColorMatchesCV(t *testing.T) {
	src := makeRGB(10, 12)
	codes := []cv.ColorConversionCode{
		cv.ColorRGB2Gray, cv.ColorBGR2Gray, cv.ColorRGB2BGR,
		cv.ColorRGB2HSV, cv.ColorRGB2HLS, cv.ColorRGB2Lab, cv.ColorRGB2YCrCb,
	}
	g := upload(t, src)
	for _, code := range codes {
		got := cudaimgproc.CvtColor(g, code).Download()
		want := cv.CvtColor(src, code)
		mustEqual(t, got, want)
	}
	// Round-trip through a stream argument to exercise the no-op path.
	got := cudaimgproc.CvtColor(g, cv.ColorRGB2Gray, cudaimgproc.NewStream()).Download()
	mustEqual(t, got, cv.CvtColor(src, cv.ColorRGB2Gray))
}

func TestSwapChannels(t *testing.T) {
	src := makeRGB(4, 4)
	g := upload(t, src)
	got := cudaimgproc.SwapChannels(g, []int{2, 1, 0}).Download()
	want := cv.CvtColor(src, cv.ColorRGB2BGR)
	mustEqual(t, got, want)
}

func TestGammaCorrectionRoundTrip(t *testing.T) {
	src := makeRGB(8, 8)
	g := upload(t, src)
	fwd := cudaimgproc.GammaCorrection(g, true)
	back := cudaimgproc.GammaCorrection(fwd, false).Download()
	orig := src
	// sRGB encode then decode should recover values within a small rounding
	// tolerance.
	for i := range orig.Data {
		d := int(orig.Data[i]) - int(back.Data[i])
		if d < -3 || d > 3 {
			t.Fatalf("gamma round-trip drift at %d: %d vs %d", i, orig.Data[i], back.Data[i])
		}
	}
}

func TestGammaCorrectionAlphaPreserved(t *testing.T) {
	src := cv.NewMat(2, 2, 4)
	for p := 0; p < 4; p++ {
		src.Data[p*4+0] = uint8(p * 10)
		src.Data[p*4+1] = uint8(p * 20)
		src.Data[p*4+2] = uint8(p * 30)
		src.Data[p*4+3] = uint8(100 + p)
	}
	g := upload(t, src)
	got := cudaimgproc.GammaCorrection(g, true).Download()
	for p := 0; p < 4; p++ {
		if got.Data[p*4+3] != src.Data[p*4+3] {
			t.Fatalf("alpha changed at %d", p)
		}
	}
}

func TestDemosaicBayer(t *testing.T) {
	// Single-channel mosaic; check output shape and that a red-site pixel
	// passes its own value through the red channel for the RG pattern.
	m := makeGray(6, 6)
	g := upload(t, m)
	out := cudaimgproc.DemosaicBayer(g, cudaimgproc.BayerRG)
	dl := out.Download()
	if dl.Channels != 3 {
		t.Fatalf("Channels = %d, want 3", dl.Channels)
	}
	// BayerRG tile [[R,G],[G,B]]: pixel (0,0) is a red site.
	if dl.Data[0] != m.Data[0] {
		t.Fatalf("red site not passed through: %d vs %d", dl.Data[0], m.Data[0])
	}
	// CvtColorBayer is an alias and must agree.
	alias := cudaimgproc.CvtColorBayer(g, cudaimgproc.BayerRG).Download()
	mustEqual(t, alias, dl)
	// Exercise the other patterns.
	for _, code := range []cudaimgproc.BayerCode{cudaimgproc.BayerBG, cudaimgproc.BayerGB, cudaimgproc.BayerGR} {
		if o := cudaimgproc.DemosaicBayer(g, code).Download(); o.Channels != 3 {
			t.Fatalf("pattern %d produced %d channels", code, o.Channels)
		}
	}
}

func TestAlphaCompOver(t *testing.T) {
	// Opaque red over opaque blue yields opaque red.
	a := cv.NewMat(1, 1, 4)
	a.Data = []uint8{255, 0, 0, 255}
	b := cv.NewMat(1, 1, 4)
	b.Data = []uint8{0, 0, 255, 255}
	ga, gb := upload(t, a), upload(t, b)
	out := cudaimgproc.AlphaComp(ga, gb, cudaimgproc.AlphaOver).Download()
	want := []uint8{255, 0, 0, 255}
	for i, w := range want {
		if out.Data[i] != w {
			t.Fatalf("AlphaOver[%d] = %d, want %d", i, out.Data[i], w)
		}
	}
	// Fully transparent source over opaque dst yields the dst.
	a.Data = []uint8{255, 255, 255, 0}
	out = cudaimgproc.AlphaComp(upload(t, a), gb, cudaimgproc.AlphaOver).Download()
	if out.Data[2] != 255 || out.Data[3] != 255 {
		t.Fatalf("transparent-over-opaque = %v", out.Data)
	}
	// Exercise the remaining operators for coverage.
	for _, op := range []cudaimgproc.AlphaCompType{
		cudaimgproc.AlphaIn, cudaimgproc.AlphaOut, cudaimgproc.AlphaAtop,
		cudaimgproc.AlphaXor, cudaimgproc.AlphaPlus,
	} {
		if o := cudaimgproc.AlphaComp(ga, gb, op).Download(); o.Channels != 4 {
			t.Fatalf("op %d bad channels", op)
		}
	}
}

func TestCalcHistAndBackProject(t *testing.T) {
	src := makeGray(16, 16)
	g := upload(t, src)
	hist := cudaimgproc.CalcHist(g)
	want := cv.CalcHist(src, 0)
	if len(hist) != len(want) {
		t.Fatalf("hist length %d want %d", len(hist), len(want))
	}
	for i := range hist {
		if hist[i] != want[i] {
			t.Fatalf("hist[%d] = %d want %d", i, hist[i], want[i])
		}
	}
	bp := cudaimgproc.CalcBackProject(g, hist).Download()
	mustEqual(t, bp, cv.CalcBackProject(src, 0, want))
}

func TestEqualizeHistMatchesCV(t *testing.T) {
	src := makeGray(16, 16)
	got := cudaimgproc.EqualizeHist(upload(t, src)).Download()
	mustEqual(t, got, cv.EqualizeHist(src))
}

func TestHistEven(t *testing.T) {
	src := makeGray(16, 16)
	g := upload(t, src)
	even := cudaimgproc.HistEven(g, 256, 0, 256)
	full := cv.CalcHist(src, 0)
	for i := range even {
		if even[i] != full[i] {
			t.Fatalf("HistEven[%d] = %d want %d", i, even[i], full[i])
		}
	}
	// Coarser binning: total counts must be conserved.
	coarse := cudaimgproc.HistEven(g, 4, 0, 256)
	total := 0
	for _, v := range coarse {
		total += v
	}
	if total != src.Total() {
		t.Fatalf("coarse HistEven total = %d want %d", total, src.Total())
	}
}

func TestHistRange(t *testing.T) {
	src := makeGray(10, 10)
	g := upload(t, src)
	got := cudaimgproc.HistRange(g, []int{0, 128, 256})
	// Compare against a direct count.
	var lo, hi int
	for _, v := range src.Data {
		if v < 128 {
			lo++
		} else {
			hi++
		}
	}
	if got[0] != lo || got[1] != hi {
		t.Fatalf("HistRange = %v want [%d %d]", got, lo, hi)
	}
}

func TestCLAHEMatchesCV(t *testing.T) {
	src := makeGray(20, 20)
	clahe := cudaimgproc.CreateCLAHE(2.0, 4)
	if clahe.GetClipLimit() != 2.0 || clahe.GetTilesGridSize() != 4 {
		t.Fatal("CLAHE getters wrong")
	}
	got := clahe.Apply(upload(t, src)).Download()
	mustEqual(t, got, cv.CLAHE(src, 2.0, 4))

	clahe.SetClipLimit(3.0)
	clahe.SetTilesGridSize(2)
	got2 := clahe.Apply(upload(t, src)).Download()
	mustEqual(t, got2, cv.CLAHE(src, 3.0, 2))
}

func TestCannyMatchesCV(t *testing.T) {
	src := cv.NewMat(30, 30, 1)
	cv.Rectangle(src, cv.Point{X: 8, Y: 8}, cv.Point{X: 22, Y: 22}, cv.NewScalar(255), -1)
	det := cudaimgproc.CreateCannyEdgeDetector(50, 150)
	got := det.Detect(upload(t, src)).Download()
	mustEqual(t, got, cv.Canny(src, 50, 150))
	det.SetLowThreshold(40)
	det.SetHighThreshold(120)
	got2 := det.Detect(upload(t, src)).Download()
	mustEqual(t, got2, cv.Canny(src, 40, 120))
}

func TestHoughLinesMatchesCV(t *testing.T) {
	src := cv.NewMat(40, 40, 1)
	cv.Line(src, cv.Point{X: 2, Y: 2}, cv.Point{X: 37, Y: 37}, cv.NewScalar(255), 1)
	det := cudaimgproc.CreateHoughLinesDetector(1.0, 0.017453292519943295, 10)
	got := det.Detect(upload(t, src))
	want := cv.HoughLines(src, 1.0, 0.017453292519943295, 10)
	if len(got) != len(want) {
		t.Fatalf("HoughLines count %d want %d", len(got), len(want))
	}
}

func TestHoughSegmentMatchesCV(t *testing.T) {
	src := cv.NewMat(40, 40, 1)
	cv.Line(src, cv.Point{X: 5, Y: 20}, cv.Point{X: 34, Y: 20}, cv.NewScalar(255), 1)
	det := cudaimgproc.CreateHoughSegmentDetector(1.0, 0.017453292519943295, 10, 10, 3)
	got := det.Detect(upload(t, src))
	want := cv.HoughLinesP(src, 1.0, 0.017453292519943295, 10, 10, 3)
	if len(got) != len(want) {
		t.Fatalf("HoughSegment count %d want %d", len(got), len(want))
	}
}

func TestHoughCirclesMatchesCV(t *testing.T) {
	src := cv.NewMat(50, 50, 1)
	cv.Circle(src, cv.Point{X: 25, Y: 25}, 12, cv.NewScalar(255), 1)
	det := cudaimgproc.CreateHoughCirclesDetector(20, 100, 20, 8, 16)
	got := det.Detect(upload(t, src))
	want := cv.HoughCircles(src, 20, 100, 20, 8, 16)
	if len(got) != len(want) {
		t.Fatalf("HoughCircles count %d want %d", len(got), len(want))
	}
}

func TestHarrisMatchesCV(t *testing.T) {
	src := makeGray(20, 20)
	h := cudaimgproc.CreateHarrisCorner(2, 3, 0.04)
	got := h.Compute(upload(t, src))
	want := cv.CornerHarris(src, 2, 3, 0.04)
	if got.Rows != want.Rows || got.Cols != want.Cols {
		t.Fatal("Harris shape mismatch")
	}
	for i := range got.Data {
		if got.Data[i] != want.Data[i] {
			t.Fatalf("Harris[%d] = %v want %v", i, got.Data[i], want.Data[i])
		}
	}
}

func TestMinEigenValCorner(t *testing.T) {
	// A bright square on a dark field produces strong corner responses at its
	// corners and near-zero response in flat regions.
	src := cv.NewMat(30, 30, 1)
	cv.Rectangle(src, cv.Point{X: 10, Y: 10}, cv.Point{X: 20, Y: 20}, cv.NewScalar(255), -1)
	mvc := cudaimgproc.CreateMinEigenValCorner(3, 3)
	resp := mvc.Compute(upload(t, src))
	corner := resp.At(10, 10)
	flat := resp.At(2, 2)
	if corner <= flat {
		t.Fatalf("corner response %v should exceed flat %v", corner, flat)
	}
	if flat < 0 {
		t.Fatalf("min-eigenvalue response should be non-negative, got %v", flat)
	}
}

func TestTemplateMatching(t *testing.T) {
	src := makeGray(24, 24)
	templ := src.Region(6, 8, 5, 5) // height, width from (y=6,x=8)
	tm := cudaimgproc.CreateTemplateMatching(cv.TmSqdiff)
	res := tm.Match(upload(t, src), upload(t, templ))
	want := cv.MatchTemplate(src, templ, cv.TmSqdiff)
	if res.Rows != want.Rows || res.Cols != want.Cols {
		t.Fatal("template result shape mismatch")
	}
	minVal, _, minX, minY, _, _ := cv.MinMaxLoc(res)
	if minX != 8 || minY != 6 {
		t.Fatalf("best match at (%d,%d), want (8,6)", minX, minY)
	}
	if minVal != 0 {
		t.Fatalf("exact patch should give 0 SQDIFF, got %v", minVal)
	}
}

func TestBilateralFilterMatchesCV(t *testing.T) {
	src := makeGray(16, 16)
	got := cudaimgproc.BilateralFilter(upload(t, src), 5, 30, 30).Download()
	mustEqual(t, got, cv.BilateralFilter(src, 5, 30, 30))
}

func TestBlendMatchesCV(t *testing.T) {
	a := makeRGB(8, 8)
	b2 := cv.NewMat(8, 8, 3)
	for i := range b2.Data {
		b2.Data[i] = uint8(255 - int(a.Data[i]))
	}
	got := cudaimgproc.Blend(upload(t, a), upload(t, b2), 0.25, 0.75).Download()
	mustEqual(t, got, cv.AddWeighted(a, 0.25, b2, 0.75, 0))
}

func TestBlendLinear(t *testing.T) {
	a := cv.NewMat(1, 2, 3)
	a.Data = []uint8{100, 100, 100, 200, 200, 200}
	b := cv.NewMat(1, 2, 3)
	b.Data = []uint8{0, 0, 0, 0, 0, 0}
	w1 := cv.NewMat(1, 2, 1)
	w1.Data = []uint8{1, 2}
	w2 := cv.NewMat(1, 2, 1)
	w2.Data = []uint8{1, 2}
	out := cudaimgproc.BlendLinear(upload(t, a), upload(t, b), upload(t, w1), upload(t, w2)).Download()
	// Equal weights: average of the two pixels.
	if out.Data[0] != 50 || out.Data[3] != 100 {
		t.Fatalf("BlendLinear = %v", out.Data)
	}
	// Zero weights leave the pixel at zero.
	wz := cv.NewMat(1, 2, 1)
	out2 := cudaimgproc.BlendLinear(upload(t, a), upload(t, b), upload(t, wz), upload(t, wz)).Download()
	if out2.Data[0] != 0 {
		t.Fatalf("zero-weight blend should be 0, got %d", out2.Data[0])
	}
}

func TestMeanShiftFiltering(t *testing.T) {
	// A uniform block must stay uniform after filtering.
	src := cv.NewMat(10, 10, 3)
	src.SetTo(120)
	out := cudaimgproc.MeanShiftFiltering(upload(t, src), 3, 20, 5, 1).Download()
	if out.Channels != 3 {
		t.Fatalf("channels = %d", out.Channels)
	}
	for i := range out.Data {
		if out.Data[i] != 120 {
			t.Fatalf("uniform block changed at %d: %d", i, out.Data[i])
		}
	}
}

func TestMeanShiftProc(t *testing.T) {
	src := makeRGB(8, 8)
	filtered, coord := cudaimgproc.MeanShiftProc(upload(t, src), 2, 25, 5, 1)
	if filtered.Channels() != 3 {
		t.Fatalf("filtered channels = %d", filtered.Channels())
	}
	c := coord.Download()
	if c.Channels != 2 {
		t.Fatalf("coord channels = %d, want 2", c.Channels)
	}
	// Converged coordinates must stay within the image bounds.
	for p := 0; p < c.Total(); p++ {
		x := int(c.Data[p*2+0])
		y := int(c.Data[p*2+1])
		if x < 0 || x >= 8 || y < 0 || y >= 8 {
			t.Fatalf("coordinate out of bounds: (%d,%d)", x, y)
		}
	}
}

func TestMeanShiftSegmentation(t *testing.T) {
	// Left half one colour, right half another. Segmentation should collapse
	// each half to its own uniform colour.
	src := cv.NewMat(8, 8, 3)
	for y := 0; y < 8; y++ {
		for x := 0; x < 8; x++ {
			i := (y*8 + x) * 3
			if x < 4 {
				src.Data[i+0], src.Data[i+1], src.Data[i+2] = 200, 30, 30
			} else {
				src.Data[i+0], src.Data[i+1], src.Data[i+2] = 30, 30, 200
			}
		}
	}
	out := cudaimgproc.MeanShiftSegmentation(upload(t, src), 3, 20, 4, 5, 1).Download()
	// Within each half all pixels share the same value.
	for y := 0; y < 8; y++ {
		leftBase := (y * 8) * 3
		for x := 1; x < 4; x++ {
			i := (y*8 + x) * 3
			if out.Data[i] != out.Data[leftBase] {
				t.Fatalf("left half not uniform at (%d,%d)", x, y)
			}
		}
	}
	// Left and right regions must differ.
	if out.Data[0] == out.Data[(7)*3] {
		t.Fatal("left and right regions collapsed together")
	}
}

func TestStreamNoop(t *testing.T) {
	s := cudaimgproc.NewStream()
	if s.Empty() {
		t.Fatal("NewStream should be non-null")
	}
	s.WaitForCompletion()
	if !s.QueryIfComplete() {
		t.Fatal("QueryIfComplete should be true")
	}
	var zero cudaimgproc.Stream
	if !zero.Empty() {
		t.Fatal("zero Stream should be null")
	}
}

func TestEmptyPanics(t *testing.T) {
	cases := map[string]func(){
		"CvtColor":     func() { cudaimgproc.CvtColor(cudaimgproc.GpuMat{}, cv.ColorRGB2Gray) },
		"Download":     func() { cudaimgproc.GpuMat{}.Download() },
		"Size":         func() { cudaimgproc.GpuMat{}.Size() },
		"CalcHist":     func() { cudaimgproc.CalcHist(cudaimgproc.GpuMat{}) },
		"UploadNil":    func() { var g cudaimgproc.GpuMat; g.Upload(nil) },
		"SwapBadOrder": func() { cudaimgproc.SwapChannels(upload(t, makeRGB(2, 2)), []int{0, 1}) },
	}
	for name, fn := range cases {
		t.Run(name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatalf("%s should have panicked", name)
				}
			}()
			fn()
		})
	}
}
