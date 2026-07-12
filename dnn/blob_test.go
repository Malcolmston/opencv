package dnn

import (
	"testing"

	cv "github.com/malcolmston/opencv"
)

func TestBlobFromImageNormalization(t *testing.T) {
	m := cv.NewMat(1, 1, 3) // 1x1 RGB pixel
	m.Set(0, 0, 0, 100)     // R
	m.Set(0, 0, 1, 150)     // G
	m.Set(0, 0, 2, 200)     // B

	blob := BlobFromImage(m, 2.0, []float64{10, 20, 30}, false)
	if !sameShape(blob.Shape, []int{1, 3, 1, 1}) {
		t.Fatalf("blob shape = %v want [1 3 1 1]", blob.Shape)
	}
	// value = scale*(pixel - mean)
	want := []float64{2 * (100 - 10), 2 * (150 - 20), 2 * (200 - 30)}
	for i, w := range want {
		if !almostEqual(blob.Data[i], w, 1e-12) {
			t.Fatalf("blob[%d] = %v want %v", i, blob.Data[i], w)
		}
	}
}

func TestBlobFromImageSwapRB(t *testing.T) {
	m := cv.NewMat(1, 1, 3)
	m.Set(0, 0, 0, 100) // R
	m.Set(0, 0, 1, 150) // G
	m.Set(0, 0, 2, 200) // B

	blob := BlobFromImage(m, 1.0, nil, true)
	// Output channel 0 should come from source B (200), channel 2 from R (100).
	if blob.Data[0] != 200 || blob.Data[1] != 150 || blob.Data[2] != 100 {
		t.Fatalf("swapRB blob = %v want [200 150 100]", blob.Data)
	}
}

func TestBlobRoundTrip(t *testing.T) {
	m := cv.NewMat(2, 2, 3)
	v := uint8(0)
	for y := 0; y < 2; y++ {
		for x := 0; x < 2; x++ {
			for c := 0; c < 3; c++ {
				m.Set(y, x, c, v*7%251)
				v++
			}
		}
	}
	blob := BlobFromImage(m, 1.0/255, []float64{0.1, 0.2, 0.3}, true)
	back := BlobToImage(blob, 0, 1.0/255, []float64{0.1, 0.2, 0.3}, true)
	if back.Rows != m.Rows || back.Cols != m.Cols || back.Channels != m.Channels {
		t.Fatalf("round-trip dims = %dx%dx%d", back.Rows, back.Cols, back.Channels)
	}
	for i := range m.Data {
		// Allow +/-1 for the round-trip rounding.
		diff := int(m.Data[i]) - int(back.Data[i])
		if diff < -1 || diff > 1 {
			t.Fatalf("round-trip byte %d: got %d want %d", i, back.Data[i], m.Data[i])
		}
	}
}
