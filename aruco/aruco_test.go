package aruco_test

import (
	"fmt"
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/aruco"
)

// pasteMarker renders marker id from dict and pastes it onto a white square
// canvas of the given size at (x, y), leaving a white quiet zone around it.
func pasteMarker(dict *aruco.Dictionary, id, side, canvas, x, y int) *cv.Mat {
	m := aruco.GenerateMarker(dict, id, side)
	c := cv.NewMat(canvas, canvas, 1)
	c.SetTo(255)
	m.CopyTo(c, y, x)
	return c
}

func TestDictionaryProperties(t *testing.T) {
	cases := []struct {
		name aruco.PredefinedDictionaryName
		bits int
		tol  int
	}{
		{aruco.Dict4x4, 4, 1},
		{aruco.Dict5x5, 5, 2},
	}
	for _, tc := range cases {
		d := aruco.GetPredefinedDictionary(tc.name)
		if d.BitsPerSide() != tc.bits {
			t.Errorf("%s: BitsPerSide=%d want %d", d.Name, d.BitsPerSide(), tc.bits)
		}
		if d.Tolerance() != tc.tol {
			t.Errorf("%s: Tolerance=%d want %d", d.Name, d.Tolerance(), tc.tol)
		}
		if d.Size() < 10 {
			t.Errorf("%s: Size=%d, want a usable family (>=10)", d.Name, d.Size())
		}
	}
	// The cache returns the same pointer on repeated calls.
	d1 := aruco.GetPredefinedDictionary(aruco.Dict4x4)
	d2 := aruco.GetPredefinedDictionary(aruco.Dict4x4)
	if d1 != d2 {
		t.Error("GetPredefinedDictionary should cache and return the same pointer")
	}
}

func TestGenerateMarkerShapeAndBorder(t *testing.T) {
	d := aruco.GetPredefinedDictionary(aruco.Dict4x4)
	const side = 60 // exactly (4+2)*10
	m := aruco.GenerateMarker(d, 0, side)
	if m.Rows != side || m.Cols != side || m.Channels != 1 {
		t.Fatalf("marker shape %dx%dx%d, want %dx%dx1", m.Rows, m.Cols, m.Channels, side, side)
	}
	// The one-cell border must be entirely black.
	cell := side / (d.BitsPerSide() + 2)
	for y := 0; y < side; y++ {
		for x := 0; x < side; x++ {
			border := x < cell || x >= side-cell || y < cell || y >= side-cell
			if border && m.At(y, x, 0) != 0 {
				t.Fatalf("border pixel (%d,%d)=%d, want 0", x, y, m.At(y, x, 0))
			}
		}
	}
}

func TestGenerateMarkerPanics(t *testing.T) {
	d := aruco.GetPredefinedDictionary(aruco.Dict4x4)
	mustPanic(t, "id out of range", func() { aruco.GenerateMarker(d, d.Size(), 60) })
	mustPanic(t, "sidePixels too small", func() { aruco.GenerateMarker(d, 0, 3) })
	mustPanic(t, "nil dictionary", func() { aruco.GenerateMarker(nil, 0, 60) })
}

func TestRoundTripAllIDs(t *testing.T) {
	for _, name := range []aruco.PredefinedDictionaryName{aruco.Dict4x4, aruco.Dict5x5} {
		d := aruco.GetPredefinedDictionary(name)
		for id := 0; id < d.Size(); id++ {
			canvas := pasteMarker(d, id, 84, 140, 30, 28)
			corners, ids := aruco.DetectMarkers(canvas, d)
			if len(ids) != 1 {
				t.Fatalf("%s id=%d: detected %d markers, want 1 (ids=%v)", d.Name, id, len(ids), ids)
			}
			if ids[0] != id {
				t.Fatalf("%s id=%d: detected id %d", d.Name, id, ids[0])
			}
			if len(corners) != 1 {
				t.Fatalf("%s id=%d: got %d corner sets", d.Name, id, len(corners))
			}
		}
	}
}

func TestDetectionRotationInvariant(t *testing.T) {
	d := aruco.GetPredefinedDictionary(aruco.Dict4x4)
	const id = 6
	const n = 140
	base := pasteMarker(d, id, 80, n, 30, 30)

	baseCorners, baseIDs := aruco.DetectMarkers(base, d)
	if len(baseIDs) != 1 || baseIDs[0] != id {
		t.Fatalf("base detection failed: ids=%v", baseIDs)
	}
	tl := baseCorners[0][0] // marker's top-left corner in the base image

	rotations := []struct {
		code cv.RotateCode
		// mapPoint maps a point in the base image to its location after rotating
		// an n-by-n image with code.
		mapPoint func(p cv.Point) cv.Point
	}{
		{cv.Rotate90CW, func(p cv.Point) cv.Point { return cv.Point{X: n - 1 - p.Y, Y: p.X} }},
		{cv.Rotate180, func(p cv.Point) cv.Point { return cv.Point{X: n - 1 - p.X, Y: n - 1 - p.Y} }},
		{cv.Rotate90CCW, func(p cv.Point) cv.Point { return cv.Point{X: p.Y, Y: n - 1 - p.X} }},
	}
	for _, r := range rotations {
		img := cv.Rotate(base, r.code)
		corners, ids := aruco.DetectMarkers(img, d)
		if len(ids) != 1 || ids[0] != id {
			t.Fatalf("rotation %d: ids=%v, want [%d]", r.code, ids, id)
		}
		// The reported top-left must track the marker's top-left through the
		// rotation, proving the corners are consistently re-ordered.
		want := r.mapPoint(tl)
		got := corners[0][0]
		if abs(got.X-want.X) > 3 || abs(got.Y-want.Y) > 3 {
			t.Errorf("rotation %d: top-left corner %v, want near %v", r.code, got, want)
		}
	}
}

func TestDetectRotatedMarker(t *testing.T) {
	// A marker rotated by an arbitrary angle (not a multiple of 90 degrees) is
	// still detected. This exercises non-axis-aligned quads and the degenerate-
	// quad guard in the unwarp step.
	d := aruco.GetPredefinedDictionary(aruco.Dict4x4)
	base := cv.NewMat(160, 160, 1)
	base.SetTo(255)
	aruco.GenerateMarker(d, 5, 90).CopyTo(base, 35, 35)
	for _, ang := range []float64{15, 30, -20} {
		m := cv.GetRotationMatrix2D(80, 80, ang, 1.0)
		rot := cv.WarpAffine(base, m, 160, 160, cv.InterLinear)
		_, ids := aruco.DetectMarkers(rot, d)
		if len(ids) != 1 || ids[0] != 5 {
			t.Errorf("angle=%v: ids=%v, want [5]", ang, ids)
		}
	}
}

func TestGetPredefinedDictionaryPanicsOnUnknown(t *testing.T) {
	mustPanic(t, "unknown dictionary", func() {
		aruco.GetPredefinedDictionary(aruco.PredefinedDictionaryName(999))
	})
}

func TestEstimatePoseDegenerate(t *testing.T) {
	// Collinear corners give a singular homography; the pose must be zero rather
	// than panicking.
	corners := [][4]cv.Point{{{X: 10, Y: 10}, {X: 20, Y: 20}, {X: 30, Y: 30}, {X: 40, Y: 40}}}
	k := [3][3]float64{{500, 0, 150}, {0, 500, 150}, {0, 0, 1}}
	rvecs, tvecs := aruco.EstimatePoseSingleMarkers(corners, 0.05, k, nil)
	if rvecs[0] != ([3]float64{}) || tvecs[0] != ([3]float64{}) {
		t.Errorf("degenerate pose: rvec=%v tvec=%v, want zeros", rvecs[0], tvecs[0])
	}
}

func TestNonMarkerQuadRejected(t *testing.T) {
	d := aruco.GetPredefinedDictionary(aruco.Dict4x4)

	solid := cv.NewMat(140, 140, 1)
	solid.SetTo(255)
	cv.Rectangle(solid, cv.Point{X: 40, Y: 40}, cv.Point{X: 100, Y: 100}, cv.NewScalar(0), cv.Filled)
	if _, ids := aruco.DetectMarkers(solid, d); len(ids) != 0 {
		t.Errorf("solid black square: got ids=%v, want none", ids)
	}

	outline := cv.NewMat(140, 140, 1)
	outline.SetTo(255)
	cv.Rectangle(outline, cv.Point{X: 40, Y: 40}, cv.Point{X: 100, Y: 100}, cv.NewScalar(0), 4)
	if _, ids := aruco.DetectMarkers(outline, d); len(ids) != 0 {
		t.Errorf("empty outlined square: got ids=%v, want none", ids)
	}
}

func TestTwoMarkersInOneImage(t *testing.T) {
	d := aruco.GetPredefinedDictionary(aruco.Dict5x5)
	canvas := cv.NewMat(180, 300, 1)
	canvas.SetTo(255)
	aruco.GenerateMarker(d, 7, 84).CopyTo(canvas, 40, 30)
	aruco.GenerateMarker(d, 19, 84).CopyTo(canvas, 40, 180)

	_, ids := aruco.DetectMarkers(canvas, d)
	if len(ids) != 2 {
		t.Fatalf("detected %d markers, want 2 (ids=%v)", len(ids), ids)
	}
	found := map[int]bool{ids[0]: true, ids[1]: true}
	if !found[7] || !found[19] {
		t.Errorf("detected ids=%v, want {7,19}", ids)
	}
}

func TestDetectOnColorImage(t *testing.T) {
	d := aruco.GetPredefinedDictionary(aruco.Dict4x4)
	gray := pasteMarker(d, 4, 84, 140, 28, 30)
	color := cv.CvtColor(gray, cv.ColorGray2RGB)
	_, ids := aruco.DetectMarkers(color, d)
	if len(ids) != 1 || ids[0] != 4 {
		t.Errorf("colour input: ids=%v, want [4]", ids)
	}
}

func TestDetectEmptyInputs(t *testing.T) {
	d := aruco.GetPredefinedDictionary(aruco.Dict4x4)
	if c, ids := aruco.DetectMarkers(nil, d); c != nil || ids != nil {
		t.Error("nil image should return nil results")
	}
	img := cv.NewMat(50, 50, 1)
	img.SetTo(255)
	if _, ids := aruco.DetectMarkers(img, nil); ids != nil {
		t.Error("nil dictionary should return nil results")
	}
}

func TestDrawDetectedMarkers(t *testing.T) {
	d := aruco.GetPredefinedDictionary(aruco.Dict4x4)
	canvas := pasteMarker(d, 2, 84, 140, 28, 30)
	corners, ids := aruco.DetectMarkers(canvas, d)
	if len(ids) != 1 {
		t.Fatalf("setup: ids=%v", ids)
	}
	out := aruco.DrawDetectedMarkers(canvas, corners, ids)
	if out.Channels != 3 || out.Rows != canvas.Rows || out.Cols != canvas.Cols {
		t.Fatalf("output shape %dx%dx%d, want %dx%dx3", out.Rows, out.Cols, out.Channels, canvas.Rows, canvas.Cols)
	}
	// A green outline must have been drawn somewhere.
	green := false
	for p := 0; p < out.Total(); p++ {
		if out.Data[p*3] == 0 && out.Data[p*3+1] == 255 && out.Data[p*3+2] == 0 {
			green = true
			break
		}
	}
	if !green {
		t.Error("expected a green outline in the drawn output")
	}
}

func TestEstimatePoseFrontoParallel(t *testing.T) {
	// A fronto-parallel, centred square of 100 px with fx=fy=500 and a 0.05 m
	// marker sits at z = fx*L/width = 500*0.05/100 = 0.25 m with no rotation.
	corners := [][4]cv.Point{{{X: 100, Y: 100}, {X: 200, Y: 100}, {X: 200, Y: 200}, {X: 100, Y: 200}}}
	k := [3][3]float64{{500, 0, 150}, {0, 500, 150}, {0, 0, 1}}
	rvecs, tvecs := aruco.EstimatePoseSingleMarkers(corners, 0.05, k, nil)
	if len(rvecs) != 1 || len(tvecs) != 1 {
		t.Fatalf("got %d rvecs / %d tvecs, want 1", len(rvecs), len(tvecs))
	}
	if math.Abs(tvecs[0][2]-0.25) > 1e-6 {
		t.Errorf("tvec z=%v, want 0.25", tvecs[0][2])
	}
	if math.Abs(tvecs[0][0]) > 1e-6 || math.Abs(tvecs[0][1]) > 1e-6 {
		t.Errorf("tvec x,y=%v,%v, want ~0", tvecs[0][0], tvecs[0][1])
	}
	rmag := math.Sqrt(rvecs[0][0]*rvecs[0][0] + rvecs[0][1]*rvecs[0][1] + rvecs[0][2]*rvecs[0][2])
	if rmag > 1e-6 {
		t.Errorf("rvec magnitude %v, want ~0 for a fronto-parallel marker", rmag)
	}
}

func TestEstimatePoseTranslationOffset(t *testing.T) {
	// Shift the square right of the principal point: tvec x should be positive,
	// z unchanged.
	corners := [][4]cv.Point{{{X: 200, Y: 100}, {X: 300, Y: 100}, {X: 300, Y: 200}, {X: 200, Y: 200}}}
	k := [3][3]float64{{500, 0, 150}, {0, 500, 150}, {0, 0, 1}}
	_, tvecs := aruco.EstimatePoseSingleMarkers(corners, 0.05, k, nil)
	if tvecs[0][0] <= 0 {
		t.Errorf("tvec x=%v, want positive for a marker right of centre", tvecs[0][0])
	}
	if math.Abs(tvecs[0][2]-0.25) > 1e-6 {
		t.Errorf("tvec z=%v, want 0.25", tvecs[0][2])
	}
}

// --- helpers ---

func mustPanic(t *testing.T, name string, fn func()) {
	t.Helper()
	defer func() {
		if recover() == nil {
			t.Errorf("%s: expected panic, got none", name)
		}
	}()
	fn()
}

func abs(v int) int {
	if v < 0 {
		return -v
	}
	return v
}

// --- examples ---

func ExampleGenerateMarker() {
	dict := aruco.GetPredefinedDictionary(aruco.Dict4x4)
	marker := aruco.GenerateMarker(dict, 0, 60)
	fmt.Printf("%dx%d, %d channel\n", marker.Rows, marker.Cols, marker.Channels)
	fmt.Printf("corner pixel (border) = %d\n", marker.At(0, 0, 0))
	// Output:
	// 60x60, 1 channel
	// corner pixel (border) = 0
}

func ExampleDetectMarkers() {
	dict := aruco.GetPredefinedDictionary(aruco.Dict4x4)

	// Render marker 9 onto a white canvas with a quiet zone.
	canvas := cv.NewMat(140, 140, 1)
	canvas.SetTo(255)
	aruco.GenerateMarker(dict, 9, 84).CopyTo(canvas, 30, 30)

	corners, ids := aruco.DetectMarkers(canvas, dict)
	fmt.Println("ids:", ids)
	fmt.Println("corners:", len(corners))
	// Output:
	// ids: [9]
	// corners: 1
}
