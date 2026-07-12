package dnn

import (
	"math"
	"testing"
)

func mustEqualData(t *testing.T, got *Tensor, wantShape []int, want []float64, tol float64) {
	t.Helper()
	if !sameShape(got.Shape, wantShape) {
		t.Fatalf("shape = %v, want %v", got.Shape, wantShape)
	}
	if len(got.Data) != len(want) {
		t.Fatalf("len = %d, want %d", len(got.Data), len(want))
	}
	for i := range want {
		if !almostEqual(got.Data[i], want[i], tol) {
			t.Fatalf("data[%d] = %v, want %v (all=%v)", i, got.Data[i], want[i], got.Data)
		}
	}
}

func TestPReLUShared(t *testing.T) {
	in := NewTensorFrom([]int{1, 4}, []float64{-2, -0.5, 0, 3})
	out := NewPReLU(NewTensorFrom([]int{1}, []float64{0.1})).Forward([]*Tensor{in})[0]
	mustEqualData(t, out, []int{1, 4}, []float64{-0.2, -0.05, 0, 3}, 1e-12)
}

func TestPReLUPerChannel(t *testing.T) {
	// [1,2,1,2]: channel 0 = {-4,2}, channel 1 = {6,-10}. slopes {0.5,0.25}.
	in := NewTensorFrom([]int{1, 2, 1, 2}, []float64{-4, 2, 6, -10})
	out := NewPReLU(NewTensorFrom([]int{2}, []float64{0.5, 0.25})).Forward([]*Tensor{in})[0]
	mustEqualData(t, out, []int{1, 2, 1, 2}, []float64{-2, 2, 6, -2.5}, 1e-12)
}

func TestELU(t *testing.T) {
	in := NewTensorFrom([]int{1, 3}, []float64{-1, 0, 2})
	out := NewELU(1).Forward([]*Tensor{in})[0]
	want := []float64{math.Exp(-1) - 1, 0, 2}
	mustEqualData(t, out, []int{1, 3}, want, 1e-12)
}

func TestMish(t *testing.T) {
	in := NewTensorFrom([]int{1, 3}, []float64{0, 1, 20})
	out := (&Mish{}).Forward([]*Tensor{in})[0]
	if !almostEqual(out.Data[0], 0, 1e-12) {
		t.Fatalf("Mish(0) = %v want 0", out.Data[0])
	}
	want1 := 1 * math.Tanh(math.Log1p(math.Exp(1)))
	if !almostEqual(out.Data[1], want1, 1e-12) {
		t.Fatalf("Mish(1) = %v want %v", out.Data[1], want1)
	}
	// For large x, tanh(softplus(x)) -> 1, so Mish(x) -> x.
	if !almostEqual(out.Data[2], 20, 1e-6) {
		t.Fatalf("Mish(20) = %v want ~20", out.Data[2])
	}
}

func TestSwishSiLU(t *testing.T) {
	in := NewTensorFrom([]int{1, 2}, []float64{0, 2})
	silu := NewSiLU().Forward([]*Tensor{in})[0]
	want := []float64{0, 2 * sigmoid(2)}
	mustEqualData(t, silu, []int{1, 2}, want, 1e-12)

	// Beta = 0 collapses to x/2.
	sw := NewSwish(0).Forward([]*Tensor{NewTensorFrom([]int{1, 1}, []float64{4})})[0]
	if !almostEqual(sw.Data[0], 2, 1e-12) {
		t.Fatalf("Swish(beta=0)(4) = %v want 2", sw.Data[0])
	}
}

func TestDropoutPassthrough(t *testing.T) {
	in := NewTensorFrom([]int{2, 2}, []float64{1, 2, 3, 4})
	out := NewDropout(0.5).Forward([]*Tensor{in})[0]
	if !out.Equal(in) {
		t.Fatalf("Dropout changed data: %v", out.Data)
	}
	out.Data[0] = 99
	if in.Data[0] == 99 {
		t.Fatalf("Dropout did not copy the backing data")
	}
}

func TestConvTranspose2D(t *testing.T) {
	in := NewTensorFrom([]int{1, 1, 2, 2}, []float64{1, 2, 3, 4})
	// kernel identity-diagonal [[1,0],[0,1]]
	w := NewTensorFrom([]int{1, 1, 2, 2}, []float64{1, 0, 0, 1})
	out := NewConvTranspose2D(w, nil, 1, 0, 1).Forward([]*Tensor{in})[0]
	want := []float64{
		1, 2, 0,
		3, 5, 2,
		0, 3, 4,
	}
	mustEqualData(t, out, []int{1, 1, 3, 3}, want, 1e-12)
}

func TestConvTranspose2DBias(t *testing.T) {
	in := NewTensorFrom([]int{1, 1, 1, 1}, []float64{5})
	w := NewTensorFrom([]int{1, 1, 1, 1}, []float64{2})
	b := NewTensorFrom([]int{1}, []float64{3})
	out := NewConvTranspose2D(w, b, 1, 0, 1).Forward([]*Tensor{in})[0]
	mustEqualData(t, out, []int{1, 1, 1, 1}, []float64{13}, 1e-12)
}

func TestGlobalAvgPool(t *testing.T) {
	in := NewTensorFrom([]int{1, 2, 2, 2}, []float64{1, 2, 3, 4, 10, 20, 30, 40})
	out := (&GlobalAvgPool{}).Forward([]*Tensor{in})[0]
	mustEqualData(t, out, []int{1, 2, 1, 1}, []float64{2.5, 25}, 1e-12)
}

func TestLRNAcross(t *testing.T) {
	in := NewTensorFrom([]int{1, 3, 1, 1}, []float64{1, 2, 3})
	l := &LRN{Size: 3, Alpha: 1, Beta: 1, K: 1, AcrossChannels: true, NormBySize: false}
	out := l.Forward([]*Tensor{in})[0]
	// c0: window {0,1} sum=5, scale=6 -> 1/6
	// c1: window {0,1,2} sum=14, scale=15 -> 2/15
	// c2: window {1,2} sum=13, scale=14 -> 3/14
	want := []float64{1.0 / 6, 2.0 / 15, 3.0 / 14}
	mustEqualData(t, out, []int{1, 3, 1, 1}, want, 1e-12)
}

func TestReshapeLayer(t *testing.T) {
	in := NewTensorFrom([]int{1, 2, 1, 2}, []float64{1, 2, 3, 4})
	out := NewReshape(2, 2).Forward([]*Tensor{in})[0]
	mustEqualData(t, out, []int{2, 2}, []float64{1, 2, 3, 4}, 1e-12)
	out.Data[0] = 99
	if in.Data[0] == 99 {
		t.Fatalf("Reshape aliased input backing")
	}
	// -1 inference.
	out2 := NewReshape(-1).Forward([]*Tensor{in})[0]
	if !sameShape(out2.Shape, []int{4}) {
		t.Fatalf("Reshape(-1) shape = %v", out2.Shape)
	}
}

func TestPermute(t *testing.T) {
	in := NewTensorFrom([]int{2, 3}, []float64{1, 2, 3, 4, 5, 6})
	out := NewPermute(1, 0).Forward([]*Tensor{in})[0]
	mustEqualData(t, out, []int{3, 2}, []float64{1, 4, 2, 5, 3, 6}, 1e-12)
}

func TestTransposeDefault(t *testing.T) {
	in := NewTensorFrom([]int{1, 2, 2}, []float64{1, 2, 3, 4})
	out := (&Transpose{}).Forward([]*Tensor{in})[0]
	mustEqualData(t, out, []int{1, 2, 2}, []float64{1, 3, 2, 4}, 1e-12)
}

func TestSlice(t *testing.T) {
	in := NewTensorFrom([]int{1, 4}, []float64{10, 11, 12, 13})
	out := NewSlice(1, 1, 3).Forward([]*Tensor{in})[0]
	mustEqualData(t, out, []int{1, 2}, []float64{11, 12}, 1e-12)

	step := (&Slice{Axis: 1, Start: 0, End: 4, Step: 2}).Forward([]*Tensor{in})[0]
	mustEqualData(t, step, []int{1, 2}, []float64{10, 12}, 1e-12)

	toEnd := NewSlice(1, 2, 0).Forward([]*Tensor{in})[0]
	mustEqualData(t, toEnd, []int{1, 2}, []float64{12, 13}, 1e-12)

	neg := NewSlice(1, -2, 0).Forward([]*Tensor{in})[0]
	mustEqualData(t, neg, []int{1, 2}, []float64{12, 13}, 1e-12)
}

func TestEltwise(t *testing.T) {
	a := NewTensorFrom([]int{1, 2}, []float64{1, 2})
	b := NewTensorFrom([]int{1, 2}, []float64{10, 20})

	sum := NewEltwise(EltwiseSum).Forward([]*Tensor{a, b})[0]
	mustEqualData(t, sum, []int{1, 2}, []float64{11, 22}, 1e-12)

	wsum := NewEltwiseSum([]float64{2, 0.5}).Forward([]*Tensor{a, b})[0]
	mustEqualData(t, wsum, []int{1, 2}, []float64{7, 14}, 1e-12)

	prod := NewEltwise(EltwiseProd).Forward([]*Tensor{a, b})[0]
	mustEqualData(t, prod, []int{1, 2}, []float64{10, 40}, 1e-12)

	c := NewTensorFrom([]int{1, 2}, []float64{10, 1})
	mx := NewEltwise(EltwiseMax).Forward([]*Tensor{a, c})[0]
	mustEqualData(t, mx, []int{1, 2}, []float64{10, 2}, 1e-12)
}

func TestPadding(t *testing.T) {
	in := NewTensorFrom([]int{1, 1, 2, 2}, []float64{1, 2, 3, 4})
	out := NewSpatialPadding(1, 0, 0, 1, 0).Forward([]*Tensor{in})[0]
	want := []float64{
		0, 0, 0,
		1, 2, 0,
		3, 4, 0,
	}
	mustEqualData(t, out, []int{1, 1, 3, 3}, want, 1e-12)

	// Fill value.
	fv := NewPadding([]int{0, 0, 0, 0}, []int{0, 0, 0, 1}, 7).Forward([]*Tensor{in})[0]
	mustEqualData(t, fv, []int{1, 1, 2, 3}, []float64{1, 2, 7, 3, 4, 7}, 1e-12)
}

func TestUpsampleNearest(t *testing.T) {
	in := NewTensorFrom([]int{1, 1, 2, 2}, []float64{1, 2, 3, 4})
	out := NewUpsampleNearest(2).Forward([]*Tensor{in})[0]
	want := []float64{
		1, 1, 2, 2,
		1, 1, 2, 2,
		3, 3, 4, 4,
		3, 3, 4, 4,
	}
	mustEqualData(t, out, []int{1, 1, 4, 4}, want, 1e-12)
}

func TestUpsampleBilinear(t *testing.T) {
	// 1x2 row [0,10], scale W=2, H=1 -> [0, 2.5, 7.5, 10] (half-pixel).
	in := NewTensorFrom([]int{1, 1, 1, 2}, []float64{0, 10})
	out := (&Upsample{ScaleH: 1, ScaleW: 2, Mode: UpsampleBilinear}).Forward([]*Tensor{in})[0]
	mustEqualData(t, out, []int{1, 1, 1, 4}, []float64{0, 2.5, 7.5, 10}, 1e-12)

	// Constant field stays constant.
	c := NewTensorFrom([]int{1, 1, 2, 2}, []float64{5, 5, 5, 5})
	cout := NewUpsampleBilinear(3).Forward([]*Tensor{c})[0]
	for _, v := range cout.Data {
		if !almostEqual(v, 5, 1e-12) {
			t.Fatalf("bilinear constant field produced %v", v)
		}
	}
}

func TestArgMax(t *testing.T) {
	in := NewTensorFrom([]int{2, 3}, []float64{1, 5, 2, 9, 0, 3})
	keep := NewArgMax(-1).Forward([]*Tensor{in})[0]
	mustEqualData(t, keep, []int{2, 1}, []float64{1, 0}, 0)

	drop := (&ArgMax{Axis: -1, KeepDims: false}).Forward([]*Tensor{in})[0]
	mustEqualData(t, drop, []int{2}, []float64{1, 0}, 0)

	// Ties resolve to the smallest index.
	tie := NewArgMax(-1).Forward([]*Tensor{NewTensorFrom([]int{1, 2}, []float64{3, 3})})[0]
	if tie.Data[0] != 0 {
		t.Fatalf("ArgMax tie = %v want 0", tie.Data[0])
	}
}

func TestNMSBoxes(t *testing.T) {
	boxes := []Box{
		{0, 0, 10, 10},
		{1, 1, 10, 10},
		{50, 50, 10, 10},
	}
	scores := []float64{0.9, 0.8, 0.7}
	keep := NMSBoxes(boxes, scores, 0.5, 0.5)
	if len(keep) != 2 || keep[0] != 0 || keep[1] != 2 {
		t.Fatalf("NMSBoxes kept %v, want [0 2]", keep)
	}
	// Raising the score threshold drops box 2, and box 1 is suppressed by 0.
	keep2 := NMSBoxes(boxes, scores, 0.75, 0.5)
	if len(keep2) != 1 || keep2[0] != 0 {
		t.Fatalf("NMSBoxes kept %v, want [0]", keep2)
	}
}

func TestBoxIoU(t *testing.T) {
	a := Box{0, 0, 10, 10}
	b := Box{1, 1, 10, 10}
	got := a.IoU(b)
	want := 81.0 / 119.0
	if !almostEqual(got, want, 1e-12) {
		t.Fatalf("IoU = %v want %v", got, want)
	}
	if a.IoU(Box{100, 100, 5, 5}) != 0 {
		t.Fatalf("disjoint IoU should be 0")
	}
}

func TestClassifyTopK(t *testing.T) {
	scores := NewTensorFrom([]int{1, 5}, []float64{0.1, 0.5, 0.2, 0.5, 0.05})
	top := ClassifyTopK(scores, 3)
	if len(top) != 3 {
		t.Fatalf("len = %d want 3", len(top))
	}
	if top[0].Index != 1 || top[1].Index != 3 || top[2].Index != 2 {
		t.Fatalf("top indices = %v", top)
	}
	if !almostEqual(top[0].Score, 0.5, 1e-12) {
		t.Fatalf("top score = %v", top[0].Score)
	}
	// Rank-1 input, k larger than classes clamps.
	flat := NewTensorFrom([]int{3}, []float64{2, 9, 4})
	all := ClassifyTopK(flat, 10)
	if len(all) != 3 || all[0].Index != 1 {
		t.Fatalf("rank-1 topk = %v", all)
	}
}

func TestNewLayersInSequential(t *testing.T) {
	// A small pipeline exercising several new single-in/single-out layers.
	net := NewSequential().
		Padding(NewSpatialPadding(0, 0, 0, 0, 0)).
		Upsample(NewUpsampleNearest(2)).
		GlobalAvgPool().
		Flatten().
		Build()
	in := NewTensorFrom([]int{1, 1, 1, 1}, []float64{4})
	out := net.Forward(in)
	// Upsample 2x replicates the single value; global-avg-pool returns it.
	mustEqualData(t, out, []int{1, 1}, []float64{4}, 1e-12)
}
