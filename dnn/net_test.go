package dnn

import "testing"

func TestNetAssemblyAndLookup(t *testing.T) {
	net := NewNet()
	net.AddNamed("act", &ReLU{})
	net.Add(&Sigmoid{}) // auto-named "layer1"

	if net.Len() != 2 {
		t.Fatalf("Len = %d want 2", net.Len())
	}
	if got := net.LayerNames(); len(got) != 2 || got[0] != "act" || got[1] != "layer1" {
		t.Fatalf("LayerNames = %v", got)
	}
	if l, ok := net.LayerByName("act"); !ok {
		t.Fatalf("LayerByName(act) not found")
	} else if _, isReLU := l.(*ReLU); !isReLU {
		t.Fatalf("LayerByName(act) returned wrong type %T", l)
	}
	if _, ok := net.LayerByName("missing"); ok {
		t.Fatalf("LayerByName(missing) should not be found")
	}
}

func TestNetDuplicateNamePanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatalf("expected panic on duplicate layer name")
		}
	}()
	NewNet().AddNamed("x", &ReLU{}).AddNamed("x", &Tanh{})
}

func TestSequentialBuildsEveryLayerKind(t *testing.T) {
	convW := NewTensor(1, 1, 1, 1)
	convW.Data[0] = 1
	fcW := NewTensor(2, 1)
	fcW.Data[0], fcW.Data[1] = 1, 1
	mean := NewTensorFrom([]int{1}, []float64{0})
	variance := NewTensorFrom([]int{1}, []float64{1})

	net := NewSequential().
		Conv2D(NewConv2D(convW, nil, 1, 0, 1)).
		BatchNorm(NewBatchNorm(nil, nil, mean, variance, 1e-5)).
		LeakyReLU(0.1).
		Sigmoid().
		Tanh().
		AvgPool2D(NewAvgPool2D(1, 1)).
		Flatten().
		Dense(NewDense(fcW, nil)).
		Softmax().
		Build()

	if net.Len() != 9 {
		t.Fatalf("Len = %d want 9", net.Len())
	}
	// A 1x1x1x1 input should flow through without panicking.
	out := net.Forward(NewTensorFrom([]int{1, 1, 1, 1}, []float64{0.5}))
	if !sameShape(out.Shape, []int{1, 2}) {
		t.Fatalf("output shape = %v want [1 2]", out.Shape)
	}
	if !almostEqual(out.Data[0]+out.Data[1], 1, 1e-12) {
		t.Fatalf("softmax output %v does not sum to 1", out.Data)
	}
}

func TestForwardMultiWithConcat(t *testing.T) {
	// A one-layer network whose layer takes two inputs (Concat).
	net := NewNet().Add(NewConcat(0))
	a := NewTensorFrom([]int{1, 2}, []float64{1, 2})
	b := NewTensorFrom([]int{1, 2}, []float64{3, 4})
	out := net.ForwardMulti([]*Tensor{a, b})
	if len(out) != 1 || !sameShape(out[0].Shape, []int{2, 2}) {
		t.Fatalf("ForwardMulti concat shape = %v", out[0].Shape)
	}
}

func TestTensorDimAndString(t *testing.T) {
	x := NewTensor(2, 3, 4)
	if x.Dim(1) != 3 {
		t.Fatalf("Dim(1) = %d want 3", x.Dim(1))
	}
	if x.Len() != 24 {
		t.Fatalf("Len = %d want 24", x.Len())
	}
	if s := x.String(); s != "Tensor[2x3x4]" {
		t.Fatalf("String = %q", s)
	}
}

func TestSoftmaxAxisZero(t *testing.T) {
	// Softmax down the columns (axis 0) of a 2x2 matrix.
	in := NewTensorFrom([]int{2, 2}, []float64{1, 1, 1, 1})
	out := NewSoftmaxAxis(0).Forward([]*Tensor{in})[0]
	for c := 0; c < 2; c++ {
		if !almostEqual(out.At(0, c)+out.At(1, c), 1, 1e-12) {
			t.Fatalf("axis-0 softmax column %d does not sum to 1", c)
		}
	}
}
