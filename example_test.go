package cv

import "fmt"

func ExampleCvtColor() {
	// A single white RGB pixel converts to full-intensity gray.
	m := NewMat(1, 1, 3)
	m.SetPixel(0, 0, []uint8{255, 255, 255})
	gray := CvtColor(m, ColorRGB2Gray)
	fmt.Println(gray.At(0, 0, 0))
	// Output: 255
}

func ExampleThreshold() {
	m := grayLine(10, 100, 200)
	out, used := Threshold(m, 120, 255, ThreshBinary)
	fmt.Printf("%v %d\n", out.Data, int(used))
	// Output: [0 0 255] 120
}

func ExampleBlur() {
	// The mean of the 3x3 neighbourhood at the centre of a ramp is 50.
	m := NewMat(3, 3, 1)
	copy(m.Data, []uint8{10, 20, 30, 40, 50, 60, 70, 80, 90})
	out := Blur(m, 3)
	fmt.Println(out.At(1, 1, 0))
	// Output: 50
}

func ExampleMatchTemplate() {
	src := NewMat(6, 6, 1)
	for i := range src.Data {
		src.Data[i] = uint8(i)
	}
	templ := src.Region(2, 3, 2, 2)
	res := MatchTemplate(src, templ, TmSqdiff)
	_, _, minX, minY, _, _ := MinMaxLoc(res)
	fmt.Printf("%d,%d\n", minX, minY)
	// Output: 3,2
}

func ExampleFindContours() {
	// A 20x20 filled square yields one external contour of area (20-1)^2.
	m := NewMat(40, 40, 1)
	for y := 10; y < 30; y++ {
		for x := 10; x < 30; x++ {
			m.Set(y, x, 0, 255)
		}
	}
	contours, _ := FindContours(m, RetrExternal, ChainApproxSimple)
	fmt.Printf("%d contours, area %.0f, corners %d\n",
		len(contours), ContourArea(contours[0]), len(contours[0]))
	// Output: 1 contours, area 361, corners 4
}

func ExampleAddWeighted() {
	a := NewMat(1, 1, 1)
	a.Set(0, 0, 0, 100)
	b := NewMat(1, 1, 1)
	b.Set(0, 0, 0, 200)
	out := AddWeighted(a, 0.5, b, 0.5, 0)
	fmt.Println(out.At(0, 0, 0))
	// Output: 150
}

func ExampleGetPerspectiveTransform() {
	// Map a quad onto a 100x100 axis-aligned square and check a corner lands.
	src := [4]Point{{20, 20}, {80, 30}, {75, 85}, {25, 75}}
	dst := [4]Point{{0, 0}, {100, 0}, {100, 100}, {0, 100}}
	m := GetPerspectiveTransform(src, dst)
	w := m[6]*80 + m[7]*30 + m[8]
	fmt.Printf("%.0f,%.0f\n", (m[0]*80+m[1]*30+m[2])/w, (m[3]*80+m[4]*30+m[5])/w)
	// Output: 100,0
}

// grayLine is a small helper for the examples.
func grayLine(vals ...uint8) *Mat {
	m := NewMat(1, len(vals), 1)
	copy(m.Data, vals)
	return m
}
