package inpaint

import "testing"

func TestMethodString(t *testing.T) {
	cases := map[Method]string{
		MethodTelea:        "Telea",
		MethodNavierStokes: "NavierStokes",
		MethodDiffusion:    "Diffusion",
		MethodCriminisi:    "Criminisi",
		MethodPatchMatch:   "PatchMatch",
	}
	for m, want := range cases {
		if got := m.String(); got != want {
			t.Fatalf("Method(%d).String() = %q, want %q", int(m), got, want)
		}
	}
}

func TestInpaintDispatch(t *testing.T) {
	img := uniformMat(11, 11, 3, 77)
	mask := centerHoleMask(11, 11, 4, 4, 3, 3)
	for _, m := range []Method{MethodTelea, MethodNavierStokes, MethodDiffusion, MethodCriminisi, MethodPatchMatch} {
		out := Inpaint(img, mask, 3, m)
		if out.At(5, 5, 0) != 77 {
			t.Fatalf("Inpaint(%s) centre = %d, want 77", m, out.At(5, 5, 0))
		}
	}
}

func TestInpaintMatBuildsMask(t *testing.T) {
	img := uniformMat(9, 9, 1, 42)
	maskMat := centerHoleMask(9, 9, 3, 3, 3, 3).ToMat()
	out := InpaintMat(img, maskMat, 2, MethodDiffusion)
	if out.At(4, 4, 0) != 42 {
		t.Fatalf("InpaintMat centre = %d, want 42", out.At(4, 4, 0))
	}
}

func TestInpaintUnknownMethodPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatalf("expected panic on unknown method")
		}
	}()
	img := uniformMat(5, 5, 1, 10)
	mask := centerHoleMask(5, 5, 2, 2, 1, 1)
	_ = Inpaint(img, mask, 1, Method(99))
}
