package calib2

import (
	"math"
	"testing"
)

func mat3MaxDiff(a, b [3][3]float64) float64 {
	var m float64
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			if d := math.Abs(a[i][j] - b[i][j]); d > m {
				m = d
			}
		}
	}
	return m
}

func TestRodriguesKnownZ90(t *testing.T) {
	r := Rodrigues([3]float64{0, 0, math.Pi / 2})
	want := [3][3]float64{{0, -1, 0}, {1, 0, 0}, {0, 0, 1}}
	if d := mat3MaxDiff(r, want); d > 1e-12 {
		t.Fatalf("Rodrigues z90 = %v, diff %g", r, d)
	}
	if !IsRotationMatrix(r, 1e-9) {
		t.Error("z90 not recognised as rotation")
	}
}

func TestRodriguesRoundTrip(t *testing.T) {
	vecs := [][3]float64{
		{0, 0, 0},
		{0.1, -0.2, 0.35},
		{1.2, 0.4, -0.7},
		{0, 0, math.Pi - 0.01},
	}
	for _, v := range vecs {
		r := Rodrigues(v)
		back := RodriguesInverse(r)
		if mat3MaxDiff(Rodrigues(back), r) > 1e-10 {
			t.Errorf("round trip failed for %v: back=%v", v, back)
		}
	}
}

func TestEulerRoundTrip(t *testing.T) {
	roll, pitch, yaw := 0.3, -0.4, 1.1
	r := EulerToRotation(roll, pitch, yaw)
	gr, gp, gy := RotationToEuler(r)
	if math.Abs(gr-roll) > 1e-10 || math.Abs(gp-pitch) > 1e-10 || math.Abs(gy-yaw) > 1e-10 {
		t.Errorf("euler round trip: got (%g,%g,%g) want (%g,%g,%g)", gr, gp, gy, roll, pitch, yaw)
	}
}

func TestEulerZ90Known(t *testing.T) {
	r := EulerToRotation(0, 0, math.Pi/2)
	want := RotationZ(math.Pi / 2)
	if mat3MaxDiff(r, want) > 1e-12 {
		t.Errorf("euler yaw90 != Rz90")
	}
}

func TestQuaternionRoundTrip(t *testing.T) {
	r := EulerToRotation(0.2, 0.5, -0.9)
	q := RotationToQuaternion(r)
	if d := math.Abs(math.Sqrt(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]) - 1); d > 1e-12 {
		t.Errorf("quaternion not unit: %g", d)
	}
	back := QuaternionToRotation(q)
	if mat3MaxDiff(r, back) > 1e-10 {
		t.Errorf("quaternion round trip failed, diff %g", mat3MaxDiff(r, back))
	}
}

func TestComposeRotations(t *testing.T) {
	a := RotationZ(0.5)
	b := RotationZ(0.3)
	got := ComposeRotations(a, b)
	want := RotationZ(0.8)
	if mat3MaxDiff(got, want) > 1e-12 {
		t.Errorf("compose rotations about same axis should add angles")
	}
}

func TestIsRotationMatrixRejectsScaled(t *testing.T) {
	bad := [3][3]float64{{2, 0, 0}, {0, 1, 0}, {0, 0, 1}}
	if IsRotationMatrix(bad, 1e-6) {
		t.Error("scaled matrix accepted as rotation")
	}
}
