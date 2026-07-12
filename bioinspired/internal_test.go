package bioinspired

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

func TestSpatialConstantToCoeff(t *testing.T) {
	if spatialConstantToCoeff(0) != 0 {
		t.Fatal("zero constant should give zero coefficient")
	}
	if spatialConstantToCoeff(-1) != 0 {
		t.Fatal("negative constant should give zero coefficient")
	}
	if got := spatialConstantToCoeff(1000); got != 0.999 {
		t.Fatalf("large constant should clamp to 0.999, got %v", got)
	}
	if got := spatialConstantToCoeff(1); math.Abs(got-math.Exp(-1)) > 1e-9 {
		t.Fatalf("unexpected coefficient %v", got)
	}
}

func TestTemporalConstantToRetention(t *testing.T) {
	if temporalConstantToRetention(0) != 0 {
		t.Fatal("zero constant should give zero retention")
	}
	if got := temporalConstantToRetention(1e6); got != 0.999 {
		t.Fatalf("large constant should clamp to 0.999, got %v", got)
	}
}

func TestClampRound(t *testing.T) {
	if clampRound(-3) != 0 {
		t.Fatal("negative should clamp to 0")
	}
	if clampRound(300) != 255 {
		t.Fatal("large should clamp to 255")
	}
	if clampRound(127.5) != 128 {
		t.Fatalf("round-to-nearest failed: %d", clampRound(127.5))
	}
}

func TestLuminanceFallbackAverage(t *testing.T) {
	// A two-channel Mat exercises the default (non 1/3) luminance branch.
	a := newFrame(2, 2)
	b := newFrame(2, 2)
	for i := range a.data {
		a.data[i] = 10
		b.data[i] = 20
	}
	lum := luminance([]*frame{a, b})
	for _, v := range lum.data {
		if v != 15 {
			t.Fatalf("expected average 15, got %v", v)
		}
	}
}

func TestSpatialLowPassPreservesMean(t *testing.T) {
	f := newFrame(8, 8)
	for i := range f.data {
		f.data[i] = 100
	}
	out := spatialLowPass(f, 0.7)
	for _, v := range out.data {
		if math.Abs(v-100) > 1e-9 {
			t.Fatalf("low-pass changed a constant field: %v", v)
		}
	}
	// a<=0 returns a copy unchanged.
	same := spatialLowPass(f, 0)
	if &same.data[0] == &f.data[0] {
		t.Fatal("expected a fresh frame")
	}
}

func TestMatToFramesEmptyPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic on empty Mat")
		}
	}()
	matToFrames(&cv.Mat{})
}

func TestNewRetinaBadSizePanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic on bad size")
		}
	}()
	NewRetina(0, 10)
}

func TestNewToneMappingBadSizePanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic on bad size")
		}
	}()
	NewRetinaFastToneMapping(-1, 10)
}

func TestTwoChannelRunLuminanceBranch(t *testing.T) {
	rows, cols := 8, 8
	m := cv.NewMat(rows, cols, 2)
	m.SetTo(80)
	r := NewRetina(rows, cols)
	r.Run(m)
	if r.GetParvo().Channels != 2 {
		t.Fatal("parvo should preserve the 2-channel input")
	}
	// Reusing the same retina with a different channel count reallocates state.
	g := cv.NewMat(rows, cols, 1)
	g.SetTo(80)
	r.Run(g)
	if r.GetParvo().Channels != 1 {
		t.Fatal("state should reallocate for the new channel count")
	}
}

func TestRunEmptyPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic on empty input")
		}
	}()
	r := NewRetina(8, 8)
	r.Run(&cv.Mat{})
}

func TestGetMagnoRAWBeforeRunPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic reading magno before run")
		}
	}()
	NewRetina(8, 8).GetMagnoRAW()
}
