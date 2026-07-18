package core

import (
	"math"
	"testing"
)

func vec3close(a, b Vec3d, tol float64) bool {
	return math.Abs(a[0]-b[0]) < tol && math.Abs(a[1]-b[1]) < tol && math.Abs(a[2]-b[2]) < tol
}

func TestRodriguesRoundTrip(t *testing.T) {
	rvec := Vec3d{0, 0, math.Pi / 2}
	m := RodriguesToMatrixd(rvec)
	// 90° about z maps (1,0,0) -> (0,1,0).
	got := m.MulVec(Vec3d{1, 0, 0})
	if !vec3close(got, Vec3d{0, 1, 0}, 1e-9) {
		t.Errorf("rotated x = %v", got)
	}
	back := MatrixToRodriguesd(m)
	if !vec3close(back, rvec, 1e-9) {
		t.Errorf("round trip = %v, want %v", back, rvec)
	}
}

func TestAffine3dTransform(t *testing.T) {
	a := NewAffine3d(Matx33dIdentity(), Vec3d{1, 2, 3})
	p := a.TransformPoint(Pt3d(10, 10, 10))
	if !p.Equals(Pt3d(11, 12, 13)) {
		t.Errorf("transform = %v", p)
	}
	inv := a.Inv()
	back := inv.TransformPoint(p)
	if math.Abs(back.X-10) > 1e-9 || math.Abs(back.Z-10) > 1e-9 {
		t.Errorf("inverse transform = %v", back)
	}
}

func TestAffine3dConcatenate(t *testing.T) {
	t1 := NewAffine3d(Matx33dIdentity(), Vec3d{1, 0, 0})
	t2 := NewAffine3d(Matx33dIdentity(), Vec3d{0, 1, 0})
	comp := t1.Concatenate(t2)
	got := comp.TransformPoint(Pt3d(0, 0, 0))
	if !got.Equals(Pt3d(1, 1, 0)) {
		t.Errorf("concatenate = %v", got)
	}
}

func TestQuatRotation(t *testing.T) {
	q := QuatdFromAxisAngle(Vec3d{0, 0, 1}, math.Pi/2)
	got := q.RotateVector(Vec3d{1, 0, 0})
	if !vec3close(got, Vec3d{0, 1, 0}, 1e-9) {
		t.Errorf("quat rotate = %v", got)
	}
	// Quaternion <-> matrix round trip.
	m := q.ToRotationMatrix()
	q2 := QuatdFromRotationMatrix(m)
	// q and q2 represent the same rotation (possibly negated).
	if math.Abs(math.Abs(q.Dot(q2))-1) > 1e-9 {
		t.Errorf("quat/matrix round trip mismatch: %v vs %v", q, q2)
	}
}

func TestQuatMulIdentity(t *testing.T) {
	q := NewQuatd(0.5, 0.5, 0.5, 0.5)
	id := QuatdIdentity()
	if got := q.Mul(id); !got.Equals(q) {
		t.Errorf("q*I = %v", got)
	}
	prod := q.Mul(q.Inverse())
	if math.Abs(prod.W-1) > 1e-9 || math.Abs(prod.X) > 1e-9 {
		t.Errorf("q*q^-1 = %v", prod)
	}
}

func TestSlerpEndpoints(t *testing.T) {
	a := QuatdIdentity()
	b := QuatdFromAxisAngle(Vec3d{0, 0, 1}, math.Pi/2)
	if got := Slerpd(a, b, 0); math.Abs(got.Dot(a)-1) > 1e-9 {
		t.Errorf("slerp(0) = %v", got)
	}
	mid := Slerpd(a, b, 0.5)
	if math.Abs(mid.Norm()-1) > 1e-9 {
		t.Errorf("slerp midpoint not unit: %v", mid.Norm())
	}
}

func TestRNGDeterministic(t *testing.T) {
	r1 := NewRNG(42)
	r2 := NewRNG(42)
	for i := 0; i < 100; i++ {
		if r1.Next() != r2.Next() {
			t.Fatal("same seed diverged")
		}
	}
	r := NewRNG(7)
	for i := 0; i < 1000; i++ {
		v := r.Uniformi(10, 20)
		if v < 10 || v >= 20 {
			t.Fatalf("uniform out of range: %d", v)
		}
	}
}

func TestRNGMT19937Deterministic(t *testing.T) {
	a := NewRNGMT19937(1)
	b := NewRNGMT19937(1)
	for i := 0; i < 50; i++ {
		if a.Next() != b.Next() {
			t.Fatal("MT diverged for same seed")
		}
	}
}

func BenchmarkRNGNext(b *testing.B) {
	r := NewRNG(1)
	for i := 0; i < b.N; i++ {
		r.Next()
	}
}
