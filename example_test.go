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
	fmt.Println(out.Data, int(used))
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

// grayLine is a small helper for the examples.
func grayLine(vals ...uint8) *Mat {
	m := NewMat(1, len(vals), 1)
	copy(m.Data, vals)
	return m
}
