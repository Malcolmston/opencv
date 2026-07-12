package datamatrix_test

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/datamatrix"
)

// ExampleEncodeText shows automatic scheme selection and a round trip through a
// rendered bitmap.
func ExampleEncodeText() {
	m, err := datamatrix.EncodeText("DATA MATRIX 2024", datamatrix.EncodeOptions{})
	if err != nil {
		panic(err)
	}
	res, err := datamatrix.DecodeText(m)
	if err != nil {
		panic(err)
	}
	fmt.Printf("%s symbol decodes to %q\n", res.SizeName, res.Text)
	// Output: 18x18 symbol decodes to "DATA MATRIX 2024"
}

// ExampleEncodeText_rectangular selects a rectangular symbol size.
func ExampleEncodeText_rectangular() {
	sym, err := datamatrix.EncodeTextSymbol("HELLO", datamatrix.EncodeOptions{
		Scheme: datamatrix.SchemeASCII,
		Size:   datamatrix.SizeRectangleOnly,
	})
	if err != nil {
		panic(err)
	}
	fmt.Printf("%s: %d rows x %d cols\n", sym.SizeName, sym.Rows, sym.Cols)
	// Output: 8x18: 8 rows x 18 cols
}

// ExampleDecodeGrid_structuredAppend round-trips a structured-append symbol.
func ExampleDecodeGrid_structuredAppend() {
	sym, _ := datamatrix.EncodeTextSymbol("SEGMENT", datamatrix.EncodeOptions{
		Append: &datamatrix.StructuredAppend{Position: 2, Total: 4, FileID: [2]byte{7, 7}},
	})
	res, _ := datamatrix.DecodeGrid(sym.Modules)
	fmt.Printf("%q part %d of %d\n", res.Text, res.Append.Position, res.Append.Total)
	// Output: "SEGMENT" part 2 of 4
}

// ExampleDecodeAll reads several symbols from one image.
func ExampleDecodeAll() {
	opts := datamatrix.Options{ModulePixels: 3, QuietZoneModules: 2, Channels: 1}
	a, _ := datamatrix.EncodeText("ALPHA", datamatrix.EncodeOptions{Scheme: datamatrix.SchemeASCII, Render: opts})
	b, _ := datamatrix.EncodeText("BETA", datamatrix.EncodeOptions{Scheme: datamatrix.SchemeASCII, Render: opts})
	canvas := cv.NewMat(a.Rows+12, a.Cols+b.Cols+24, 1)
	canvas.SetTo(255)
	a.CopyTo(canvas, 6, 6)
	b.CopyTo(canvas, 6, a.Cols+18)
	results, _ := datamatrix.DecodeAll(canvas)
	fmt.Printf("found %d symbols\n", len(results))
	// Output: found 2 symbols
}
