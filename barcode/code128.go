package barcode

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
)

// This file implements Code 128 (code set B: ASCII 32-126) with a matched
// encoder ([EncodeCode128]) and scanline decoder ([DecodeCode128]). A Code 128
// symbol is a start symbol, the data symbols, a modulo-103 checksum symbol and a
// stop symbol, each an 11-module bar/space group (the stop is 13 modules). Every
// symbol is six alternating bar/space runs whose widths (1-4 modules) sum to 11,
// listed in code128Widths below.

// code128Widths gives the bar/space run widths of Code 128 symbol values 0-106.
// Values 0-102 are data, 103-105 are the start symbols (A/B/C) and 106 is the
// stop symbol (seven runs). Each rune is a width in modules; the first run is a
// bar.
var code128Widths = [...]string{
	"212222", "222122", "222221", "121223", "121322", "131222", "122213", "122312",
	"132212", "221213", "221312", "231212", "112232", "122132", "122231", "113222",
	"123122", "123221", "223211", "221132", "221231", "213212", "223112", "312131",
	"311222", "321122", "321221", "312212", "322112", "322211", "212123", "212321",
	"232121", "111323", "131123", "131321", "112313", "132113", "132311", "211313",
	"231113", "231311", "112133", "112331", "132131", "113123", "113321", "133121",
	"313121", "211331", "231131", "213113", "213311", "213131", "311123", "311321",
	"331121", "312113", "312311", "332111", "314111", "221411", "431111", "111224",
	"111422", "121124", "121421", "141122", "141221", "112214", "112412", "122114",
	"122411", "142112", "142211", "241211", "221114", "413111", "241112", "134111",
	"111242", "121142", "121241", "114212", "124112", "124211", "411212", "421112",
	"421211", "212141", "214121", "412121", "111143", "111341", "131141", "114113",
	"114311", "411113", "411311", "113141", "114131", "311141", "411131", "211412",
	"211214", "211232", "2331112",
}

const (
	code128StartB = 104
	code128Stop   = 106

	c128ModuleWidth = 3
	c128Height      = 60
	c128Quiet       = 10
)

// widthsToModules appends the modules of a symbol given as a width string
// (first run a bar) to m.
func widthsToModules(m []bool, widths string) []bool {
	bar := true
	for _, ch := range widths {
		n := int(ch - '0')
		for i := 0; i < n; i++ {
			m = append(m, bar)
		}
		bar = !bar
	}
	return m
}

// EncodeCode128 renders text as a Code 128 (code set B) barcode and returns it
// as a single-channel grayscale [cv.Mat] (bars 0, spaces 255) with quiet zones.
// text must consist of printable ASCII in the range 32-126; it returns an error
// otherwise.
func EncodeCode128(text string) (*cv.Mat, error) {
	values := []int{code128StartB}
	sum := code128StartB
	for i := 0; i < len(text); i++ {
		c := text[i]
		if c < 32 || c > 126 {
			return nil, fmt.Errorf("barcode: Code 128 set B cannot encode byte %d", c)
		}
		v := int(c) - 32
		values = append(values, v)
		sum += v * (i + 1)
	}
	values = append(values, sum%103, code128Stop)

	var modules []bool
	for _, v := range values {
		modules = widthsToModules(modules, code128Widths[v])
	}
	totalModules := len(modules) + 2*c128Quiet
	w := totalModules * c128ModuleWidth
	m := cv.NewMat(c128Height, w, 1)
	m.SetTo(255)
	for i, bar := range modules {
		if !bar {
			continue
		}
		x0 := (i + c128Quiet) * c128ModuleWidth
		for y := 0; y < c128Height; y++ {
			for dx := 0; dx < c128ModuleWidth; dx++ {
				m.Set(y, x0+dx, 0, 0)
			}
		}
	}
	return m, nil
}

// runWidths converts a slice of modules into its bar/space run-length widths.
func runWidths(mod []bool) []int {
	if len(mod) == 0 {
		return nil
	}
	var out []int
	cur := mod[0]
	count := 0
	for _, b := range mod {
		if b == cur {
			count++
		} else {
			out = append(out, count)
			cur = b
			count = 1
		}
	}
	out = append(out, count)
	return out
}

// matchSymbol returns the Code 128 value whose width pattern best fits the given
// run widths, or -1 if the run count is wrong.
func matchSymbol(widths []int) int {
	best, bestErr := -1, 1<<31-1
	for v, pat := range code128Widths {
		if len(pat) != len(widths) {
			continue
		}
		err := 0
		for i := 0; i < len(pat); i++ {
			d := int(pat[i]-'0') - widths[i]
			if d < 0 {
				d = -d
			}
			err += d
		}
		if err < bestErr {
			bestErr = err
			best = v
		}
	}
	return best
}

// DecodeCode128 scans a rendered Code 128 (code set B) barcode and returns the
// decoded text and true on success, or ("", false) if no valid symbol is found.
// It reads a middle scanline, locates the barcode by its dark extent, samples
// the modules, splits them into 11-module symbols plus the 13-module stop,
// matches each symbol's run widths, and verifies the modulo-103 checksum.
func DecodeCode128(img *cv.Mat) (string, bool) {
	if img == nil || img.Empty() {
		return "", false
	}
	gray := img
	if img.Channels != 1 {
		gray = cv.CvtColor(img, cv.ColorRGB2Gray)
	}
	bin, _ := cv.Threshold(gray, 0, 255, cv.ThreshBinaryInv|cv.ThreshOtsu)
	w := bin.Cols
	y := bin.Rows / 2
	first, last := -1, -1
	for x := 0; x < w; x++ {
		if bin.Data[y*w+x] != 0 {
			if first < 0 {
				first = x
			}
			last = x
		}
	}
	if first < 0 {
		return "", false
	}
	span := last - first + 1
	// Estimate the module width as the narrowest bar/space run (one module) in
	// pixels, then the total module count. Total modules = k 11-module symbols
	// (start + data + checksum) plus a 13-module stop symbol.
	pxRuns := runWidths(binRow(bin, y, first, last))
	minRun := span
	for _, r := range pxRuns {
		if r < minRun {
			minRun = r
		}
	}
	if minRun <= 0 {
		return "", false
	}
	total := int(float64(span)/float64(minRun) + 0.5)
	if (total-13)%11 != 0 {
		return "", false
	}
	k := (total - 13) / 11
	if k < 3 {
		return "", false
	}
	mod := make([]bool, total)
	for i := 0; i < total; i++ {
		x := first + int((float64(i)+0.5)*float64(span)/float64(total))
		mod[i] = bin.Data[y*w+x] != 0
	}
	values := make([]int, 0, k+1)
	for j := 0; j < k; j++ {
		v := matchSymbol(runWidths(mod[j*11 : (j+1)*11]))
		if v < 0 {
			return "", false
		}
		values = append(values, v)
	}
	if matchSymbol(runWidths(mod[k*11:])) != code128Stop {
		return "", false
	}
	if values[0] != code128StartB {
		return "", false
	}
	sum := code128StartB
	for i := 1; i < k-1; i++ {
		sum += values[i] * i
	}
	if sum%103 != values[k-1] {
		return "", false
	}
	out := make([]byte, 0, k-2)
	for i := 1; i < k-1; i++ {
		out = append(out, byte(values[i]+32))
	}
	return string(out), true
}

// binRow returns the boolean dark values of row y from column lo to hi.
func binRow(bin *cv.Mat, y, lo, hi int) []bool {
	w := bin.Cols
	out := make([]bool, 0, hi-lo+1)
	for x := lo; x <= hi; x++ {
		out = append(out, bin.Data[y*w+x] != 0)
	}
	return out
}
