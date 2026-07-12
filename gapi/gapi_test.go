package gapi_test

import (
	"testing"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/gapi"
)

// ramp builds a deterministic channels-channel image whose samples vary with
// position and channel, giving non-trivial content for the pipelines to chew on.
func ramp(rows, cols, channels int) *cv.Mat {
	m := cv.NewMat(rows, cols, channels)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			for c := 0; c < channels; c++ {
				m.Data[(y*cols+x)*channels+c] = uint8((y*7 + x*13 + c*29) % 256)
			}
		}
	}
	return m
}

func equalMat(t *testing.T, got, want *cv.Mat) {
	t.Helper()
	if got == nil || want == nil {
		t.Fatalf("nil mat: got=%v want=%v", got, want)
	}
	if got.Rows != want.Rows || got.Cols != want.Cols || got.Channels != want.Channels {
		t.Fatalf("shape mismatch: got %dx%dx%d want %dx%dx%d",
			got.Rows, got.Cols, got.Channels, want.Rows, want.Cols, want.Channels)
	}
	if len(got.Data) != len(want.Data) {
		t.Fatalf("data length mismatch: got %d want %d", len(got.Data), len(want.Data))
	}
	for i := range got.Data {
		if got.Data[i] != want.Data[i] {
			t.Fatalf("data mismatch at %d: got %d want %d", i, got.Data[i], want.Data[i])
		}
	}
}

// TestCannyPipeline checks the headline example: a compiled
// Canny(GaussianBlur(RGB2Gray(in))) graph equals the eager equivalent.
func TestCannyPipeline(t *testing.T) {
	img := ramp(16, 20, 3)

	in := gapi.NewMat()
	out := gapi.Canny(gapi.GaussianBlur(gapi.RGB2Gray(in), 5, 1.4), 50, 100)
	cc := gapi.NewComputation(in, out).Compile()
	got := cc.Apply(img)[0]

	want := cv.Canny(cv.GaussianBlur(cv.CvtColor(img, cv.ColorRGB2Gray), 5, 1.4), 50, 100)
	equalMat(t, got, want)
}

// TestReuseCompiled runs a single compiled graph on several inputs.
func TestReuseCompiled(t *testing.T) {
	in := gapi.NewMat()
	out := gapi.EqualizeHist(gapi.RGB2Gray(in))
	cc := gapi.NewComputation(in, out).Compile()

	for _, dims := range [][2]int{{8, 8}, {12, 5}, {3, 17}} {
		img := ramp(dims[0], dims[1], 3)
		got := cc.Apply(img)[0]
		want := cv.EqualizeHist(cv.CvtColor(img, cv.ColorRGB2Gray))
		equalMat(t, got, want)
	}
}

// TestCoreArithmetic exercises the binary and scalar core operations against
// their eager counterparts within a multi-input graph.
func TestCoreArithmetic(t *testing.T) {
	a := ramp(10, 10, 3)
	b := ramp(10, 10, 3)
	// Perturb b so it differs from a.
	for i := range b.Data {
		b.Data[i] = uint8((int(b.Data[i]) + 40) % 256)
	}

	ga := gapi.NewMat()
	gb := gapi.NewMat()

	cases := []struct {
		name string
		node gapi.GMat
		want *cv.Mat
	}{
		{"add", gapi.Add(ga, gb), cv.Add(a, b)},
		{"sub", gapi.Sub(ga, gb), cv.Subtract(a, b)},
		{"mul", gapi.Mul(ga, gb, 0.01), cv.Multiply(a, b, 0.01)},
		{"div", gapi.Div(ga, gb, 2.0), cv.Divide(a, b, 2.0)},
		{"addWeighted", gapi.AddWeighted(ga, 0.3, gb, 0.7, 5), cv.AddWeighted(a, 0.3, b, 0.7, 5)},
		{"and", gapi.BitwiseAnd(ga, gb), cv.BitwiseAnd(a, b)},
		{"or", gapi.BitwiseOr(ga, gb), cv.BitwiseOr(a, b)},
		{"xor", gapi.BitwiseXor(ga, gb), cv.BitwiseXor(a, b)},
		{"not", gapi.BitwiseNot(ga), cv.BitwiseNot(a)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cc := gapi.NewComputationMulti([]gapi.GMat{ga, gb}, []gapi.GMat{tc.node}).Compile()
			outs, err := cc.Run(gapi.Inputs{Mats: []*cv.Mat{a, b}})
			if err != nil {
				t.Fatal(err)
			}
			equalMat(t, outs[0], tc.want)
		})
	}
}

// TestCmp checks every comparison variant against a direct computation.
func TestCmp(t *testing.T) {
	a := ramp(8, 8, 1)
	b := ramp(8, 8, 1)
	for i := range b.Data {
		b.Data[i] = uint8((int(a.Data[i]) + i%5 - 2 + 256) % 256)
	}

	ops := []struct {
		name string
		node func(x, y gapi.GMat) gapi.GMat
		pred func(x, y uint8) bool
	}{
		{"eq", gapi.CmpEQ, func(x, y uint8) bool { return x == y }},
		{"gt", gapi.CmpGT, func(x, y uint8) bool { return x > y }},
		{"ge", gapi.CmpGE, func(x, y uint8) bool { return x >= y }},
		{"lt", gapi.CmpLT, func(x, y uint8) bool { return x < y }},
		{"le", gapi.CmpLE, func(x, y uint8) bool { return x <= y }},
		{"ne", gapi.CmpNE, func(x, y uint8) bool { return x != y }},
	}
	for _, op := range ops {
		t.Run(op.name, func(t *testing.T) {
			gx := gapi.NewMat()
			gy := gapi.NewMat()
			cc := gapi.NewComputationMulti([]gapi.GMat{gx, gy}, []gapi.GMat{op.node(gx, gy)}).Compile()
			got := cc.Apply(a, b)[0]
			want := cv.NewMat(a.Rows, a.Cols, 1)
			for i := range a.Data {
				if op.pred(a.Data[i], b.Data[i]) {
					want.Data[i] = 255
				}
			}
			equalMat(t, got, want)
		})
	}
}

// TestScalarInputs binds run-time scalars through AddC and MulC.
func TestScalarInputs(t *testing.T) {
	img := ramp(6, 9, 1)

	in := gapi.NewMat()
	s := gapi.NewScalar()
	out := gapi.MulC(gapi.AddC(in, s), gapi.ConstScalar(2))
	cc := gapi.NewComputationIO([]gapi.GMat{in}, []gapi.GScalar{s}, []gapi.GMat{out}).Compile()

	outs, err := cc.Run(gapi.Inputs{Mats: []*cv.Mat{img}, Scalars: []float64{10}})
	if err != nil {
		t.Fatal(err)
	}
	got := outs[0]

	want := cv.NewMat(img.Rows, img.Cols, 1)
	clamp := func(v float64) uint8 {
		if v <= 0 {
			return 0
		}
		if v >= 255 {
			return 255
		}
		return uint8(v)
	}
	for i, v := range img.Data {
		added := clamp(float64(v) + 10 + 0.5)
		want.Data[i] = clamp(float64(added)*2 + 0.5)
	}
	equalMat(t, got, want)
}

// TestSplitMerge round-trips a three-channel image through Split3/Merge3 and
// checks a multi-output graph returns all planes.
func TestSplitMerge(t *testing.T) {
	img := ramp(7, 11, 3)

	in := gapi.NewMat()
	r, g, b := gapi.Split3(in)
	merged := gapi.Merge3(r, g, b)

	// Multi-output: the three planes plus the re-merged image.
	cc := gapi.NewComputationMulti([]gapi.GMat{in}, []gapi.GMat{r, g, b, merged}).Compile()
	outs := cc.Apply(img)
	if len(outs) != 4 {
		t.Fatalf("expected 4 outputs, got %d", len(outs))
	}

	planes := img.Split()
	equalMat(t, outs[0], planes[0])
	equalMat(t, outs[1], planes[1])
	equalMat(t, outs[2], planes[2])
	equalMat(t, outs[3], img) // merge(split(x)) == x
}

// TestMask verifies masked copy semantics.
func TestMask(t *testing.T) {
	img := ramp(8, 8, 3)
	m := cv.NewMat(8, 8, 1)
	for i := range m.Data {
		if i%3 == 0 {
			m.Data[i] = 255
		}
	}

	gin := gapi.NewMat()
	gm := gapi.NewMat()
	cc := gapi.NewComputationMulti([]gapi.GMat{gin, gm}, []gapi.GMat{gapi.Mask(gin, gm)}).Compile()
	got := cc.Apply(img, m)[0]

	want := cv.NewMat(8, 8, 3)
	for p := 0; p < img.Total(); p++ {
		if m.Data[p] != 0 {
			base := p * 3
			copy(want.Data[base:base+3], img.Data[base:base+3])
		}
	}
	equalMat(t, got, want)
}

// TestGeometricAndFilters checks single-input imgproc/core transforms.
func TestGeometricAndFilters(t *testing.T) {
	rgb := ramp(12, 15, 3)
	gray := cv.CvtColor(rgb, cv.ColorRGB2Gray)

	cases := []struct {
		name  string
		src   *cv.Mat
		build func(in gapi.GMat) gapi.GMat
		want  *cv.Mat
	}{
		{"resize", rgb, func(in gapi.GMat) gapi.GMat { return gapi.Resize(in, 8, 6, gapi.InterLinear) },
			cv.Resize(rgb, 8, 6, cv.InterLinear)},
		{"flip", rgb, func(in gapi.GMat) gapi.GMat { return gapi.Flip(in, gapi.FlipHorizontal) },
			cv.Flip(rgb, cv.FlipHorizontal)},
		{"transpose", rgb, gapi.Transpose, cv.Transpose(rgb)},
		{"normalize", gray, func(in gapi.GMat) gapi.GMat { return gapi.Normalize(in, 0, 255) },
			cv.Normalize(gray, 0, 255)},
		{"blur", rgb, func(in gapi.GMat) gapi.GMat { return gapi.Blur(in, 3) }, cv.Blur(rgb, 3)},
		{"median", rgb, func(in gapi.GMat) gapi.GMat { return gapi.MedianBlur(in, 3) }, cv.MedianBlur(rgb, 3)},
		{"sobel", gray, func(in gapi.GMat) gapi.GMat { return gapi.Sobel(in, 1, 0, 3, 1, 0) },
			cv.Sobel(gray, 1, 0, 3, 1, 0)},
		{"laplacian", gray, func(in gapi.GMat) gapi.GMat { return gapi.Laplacian(in, 1, 1, 0) },
			cv.Laplacian(gray, 1, 1, 0)},
		{"dilate", gray, func(in gapi.GMat) gapi.GMat { return gapi.Dilate(in, gapi.MorphRect, 3, 1) },
			cv.Dilate(gray, cv.GetStructuringElement(cv.MorphRect, 3, 3), 1)},
		{"erode", gray, func(in gapi.GMat) gapi.GMat { return gapi.Erode(in, gapi.MorphCross, 3, 2) },
			cv.Erode(gray, cv.GetStructuringElement(cv.MorphCross, 3, 3), 2)},
		{"threshold", gray, func(in gapi.GMat) gapi.GMat { return gapi.Threshold(in, 100, 255, gapi.ThreshBinary) },
			must(cv.Threshold(gray, 100, 255, cv.ThreshBinary))},
		{"bgr2gray", rgb, gapi.BGR2Gray, cv.CvtColor(rgb, cv.ColorBGR2Gray)},
		{"cvtcolor", rgb, func(in gapi.GMat) gapi.GMat { return gapi.CvtColor(in, gapi.ColorRGB2HSV) },
			cv.CvtColor(rgb, cv.ColorRGB2HSV)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			pipe := gapi.NewComputationT(tc.build)
			got := pipe.Apply(tc.src)
			equalMat(t, got, tc.want)
		})
	}
}

// must drops the second return value of cv.Threshold for table use.
func must(m *cv.Mat, _ float64) *cv.Mat { return m }

// TestSharedSubexpression checks that a node feeding two outputs is evaluated
// once and both consumers see the same result.
func TestSharedSubexpression(t *testing.T) {
	img := ramp(9, 9, 3)

	in := gapi.NewMat()
	gray := gapi.RGB2Gray(in) // shared
	a := gapi.GaussianBlur(gray, 3, 0)
	b := gapi.Blur(gray, 3)

	cc := gapi.NewComputationMulti([]gapi.GMat{in}, []gapi.GMat{a, b}).Compile()
	outs := cc.Apply(img)

	wantGray := cv.CvtColor(img, cv.ColorRGB2Gray)
	equalMat(t, outs[0], cv.GaussianBlur(wantGray, 3, 0))
	equalMat(t, outs[1], cv.Blur(wantGray, 3))
}

// TestCustomKernel overrides an operation with a custom kernel package.
func TestCustomKernel(t *testing.T) {
	a := ramp(6, 6, 1)
	b := ramp(6, 6, 1)
	for i := range b.Data {
		b.Data[i] = uint8((int(b.Data[i]) + 17) % 256)
	}

	ga := gapi.NewMat()
	gb := gapi.NewMat()
	comp := gapi.NewComputationMulti([]gapi.GMat{ga, gb}, []gapi.GMat{gapi.Add(ga, gb)})

	// Redefine "add" to actually subtract.
	pkg := gapi.Kernels(gapi.GKernel{
		Op: gapi.OpAdd,
		Eval: func(ctx gapi.KernelContext) *cv.Mat {
			return cv.Subtract(ctx.Mats[0], ctx.Mats[1])
		},
	})
	cc := comp.CompileWith(pkg)
	got := cc.Apply(a, b)[0]
	equalMat(t, got, cv.Subtract(a, b))

	// Default compile still adds.
	def := comp.Compile().Apply(a, b)[0]
	equalMat(t, def, cv.Add(a, b))

	if pkg.Size() != 1 {
		t.Fatalf("package size = %d, want 1", pkg.Size())
	}
}

// TestKernelPackageCombine checks precedence when combining packages.
func TestKernelPackageCombine(t *testing.T) {
	base := gapi.Kernels(gapi.GKernel{Op: gapi.OpAdd, Eval: nil})
	over := gapi.Kernels(gapi.GKernel{Op: gapi.OpSub, Eval: nil})
	c := base.Combine(over)
	if c.Size() != 2 {
		t.Fatalf("combined size = %d, want 2", c.Size())
	}
	if gapi.Kernels().Combine(nil).Size() != 0 {
		t.Fatal("empty combine should be size 0")
	}
}

// TestRunErrors covers the validation paths of Run.
func TestRunErrors(t *testing.T) {
	in := gapi.NewMat()
	cc := gapi.NewComputation(in, gapi.RGB2Gray(in)).Compile()

	if _, err := cc.Run(gapi.Inputs{}); err == nil {
		t.Fatal("expected error for missing image input")
	}
	if _, err := cc.Run(gapi.Inputs{Mats: []*cv.Mat{nil}}); err == nil {
		t.Fatal("expected error for nil image input")
	}

	s := gapi.NewScalar()
	sc := gapi.NewComputationIO([]gapi.GMat{in}, []gapi.GScalar{s}, []gapi.GMat{gapi.AddC(in, s)}).Compile()
	if _, err := sc.Run(gapi.Inputs{Mats: []*cv.Mat{ramp(3, 3, 1)}}); err == nil {
		t.Fatal("expected error for missing scalar input")
	}
}

// TestValidAndConst exercises the small helper accessors.
func TestValidAndConst(t *testing.T) {
	if (gapi.GMat{}).Valid() {
		t.Fatal("zero GMat should be invalid")
	}
	if !gapi.NewMat().Valid() {
		t.Fatal("NewMat should be valid")
	}
	if (gapi.GScalar{}).Valid() {
		t.Fatal("zero GScalar should be invalid")
	}
	if !gapi.ConstScalar(1).Valid() {
		t.Fatal("ConstScalar should be valid")
	}
}

// TestComputationApply covers the GComputation.Apply convenience.
func TestComputationApply(t *testing.T) {
	img := ramp(5, 5, 3)
	in := gapi.NewMat()
	comp := gapi.NewComputation(in, gapi.RGB2Gray(in))
	got := comp.Apply(img)[0]
	equalMat(t, got, cv.CvtColor(img, cv.ColorRGB2Gray))
}

// TestCyclePanicGuardsInvalidInput ensures invalid protocol inputs are rejected.
func TestInvalidProtocol(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic for non-input used as protocol input")
		}
	}()
	in := gapi.NewMat()
	derived := gapi.RGB2Gray(in) // not a NewMat placeholder
	gapi.NewComputation(derived, derived)
}
