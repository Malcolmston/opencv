package cv

import "fmt"

// ExampleInvert inverts a 2×2 matrix and confirms M·M⁻¹ is the identity.
func ExampleInvert() {
	m := NewFloatMat(2, 2)
	copy(m.Data, []float64{4, 7, 2, 6})
	inv, ok := Invert(m)
	prod := Gemm(m, inv, 1, nil, 0, false, false)
	fmt.Println(ok)
	fmt.Printf("%.0f %.0f %.0f %.0f\n", prod.Data[0], prod.Data[1], prod.Data[2], prod.Data[3])
	// Output:
	// true
	// 1 0 0 1
}

// ExampleReduce sums each column of a matrix into a single row.
func ExampleReduce() {
	m := NewFloatMat(2, 3)
	copy(m.Data, []float64{1, 2, 3, 4, 5, 6})
	row := Reduce(m, true, ReduceSum)
	fmt.Printf("%.0f %.0f %.0f\n", row.Data[0], row.Data[1], row.Data[2])
	// Output:
	// 5 7 9
}

// ExampleDCT shows that IDCT inverts DCT.
func ExampleDCT() {
	m := NewFloatMat(1, 4)
	copy(m.Data, []float64{1, 2, 3, 4})
	back := IDCT(DCT(m))
	fmt.Printf("%.0f %.0f %.0f %.0f\n", back.Data[0], back.Data[1], back.Data[2], back.Data[3])
	// Output:
	// 1 2 3 4
}

// ExamplePointPolygonTest classifies points against a square.
func ExamplePointPolygonTest() {
	sq := []Point{{0, 0}, {10, 0}, {10, 10}, {0, 10}}
	fmt.Println(PointPolygonTest(sq, Point{5, 5}, false))
	fmt.Println(PointPolygonTest(sq, Point{20, 5}, false))
	fmt.Println(PointPolygonTest(sq, Point{0, 5}, false))
	// Output:
	// 1
	// -1
	// 0
}

// ExampleApplyColorMap maps a grayscale ramp to RGB and reports the channel
// count of the result.
func ExampleApplyColorMap() {
	src := NewMat(1, 3, 1)
	src.Data = []uint8{0, 128, 255}
	out := ApplyColorMap(src, ColormapJet)
	fmt.Println(out.Channels)
	// Output:
	// 3
}
