package calib2

import (
	"math"
	"testing"
)

func TestVec3Ops(t *testing.T) {
	a := [3]float64{1, 2, 3}
	b := [3]float64{4, 5, 6}
	if Vec3Dot(a, b) != 32 {
		t.Errorf("dot = %g want 32", Vec3Dot(a, b))
	}
	c := Vec3Cross(a, b)
	want := [3]float64{-3, 6, -3}
	if c != want {
		t.Errorf("cross = %v want %v", c, want)
	}
	if math.Abs(Vec3Norm([3]float64{3, 4, 0})-5) > 1e-12 {
		t.Errorf("norm wrong")
	}
	n := Vec3Normalize([3]float64{0, 0, 2})
	if n != ([3]float64{0, 0, 1}) {
		t.Errorf("normalize = %v", n)
	}
}

func TestMat3MulKnown(t *testing.T) {
	a := [3][3]float64{{1, 0, 0}, {0, 2, 0}, {0, 0, 3}}
	b := [3][3]float64{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}
	c := Mat3Mul(a, b)
	want := [3][3]float64{{1, 2, 3}, {8, 10, 12}, {21, 24, 27}}
	if mat3MaxDiff(c, want) > 1e-12 {
		t.Errorf("Mat3Mul = %v want %v", c, want)
	}
}

func TestMat3Inverse(t *testing.T) {
	a := [3][3]float64{{2, 0, 0}, {0, 4, 0}, {1, 0, 1}}
	inv, ok := Mat3Inverse(a)
	if !ok {
		t.Fatal("not invertible")
	}
	prod := Mat3Mul(a, inv)
	if mat3MaxDiff(prod, Mat3Identity()) > 1e-12 {
		t.Errorf("A·A⁻¹ != I: %v", prod)
	}
}

func TestMat3InverseSingular(t *testing.T) {
	a := [3][3]float64{{1, 2, 3}, {2, 4, 6}, {0, 0, 0}}
	if _, ok := Mat3Inverse(a); ok {
		t.Error("singular reported invertible")
	}
}

func TestMat3Det(t *testing.T) {
	a := [3][3]float64{{6, 1, 1}, {4, -2, 5}, {2, 8, 7}}
	if math.Abs(Mat3Det(a)-(-306)) > 1e-9 {
		t.Errorf("det = %g want -306", Mat3Det(a))
	}
}
