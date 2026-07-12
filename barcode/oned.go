package barcode

import (
	cv "github.com/malcolmston/opencv"
)

// This file holds machinery shared by the linear (1D) symbologies added in this
// package: a common rendering routine, a scanline module recoverer that turns a
// rendered symbol back into its exact module bit array, and a narrow/wide
// element classifier used by the two-width symbologies (Code 39, Codabar,
// Interleaved 2 of 5, Code 11). Every symbol is rendered as a single-channel
// grayscale [cv.Mat] with dark bars at 0 and light spaces at 255, surrounded by
// a light quiet zone, exactly like the pre-existing EAN-13 and Code 128 code.

// Shared 1D rendering parameters. Wide elements of the two-width symbologies are
// drawn as onedWide narrow modules, giving a clean 3:1 ratio that the scanline
// recoverer can quantise unambiguously.
const (
	onedModuleWidth = 3
	onedHeight      = 60
	onedQuiet       = 10
	onedWide        = 3
)

// renderModules1D renders a module bit slice (true = dark bar, false = light
// space) as a grayscale Mat with quiet zones, each module scaled to
// onedModuleWidth pixels.
func renderModules1D(modules []bool) *cv.Mat {
	total := len(modules) + 2*onedQuiet
	w := total * onedModuleWidth
	m := cv.NewMat(onedHeight, w, 1)
	m.SetTo(255)
	for i, bar := range modules {
		if !bar {
			continue
		}
		x0 := (i + onedQuiet) * onedModuleWidth
		for y := 0; y < onedHeight; y++ {
			for dx := 0; dx < onedModuleWidth; dx++ {
				m.Set(y, x0+dx, 0, 0)
			}
		}
	}
	return m
}

// binarizeMiddleRow reduces img to a bi-level middle scanline and returns the
// binary Mat, the scan row and the dark extent [first, last]. It reports false
// when the image is empty or holds no dark pixels on that row.
func binarizeMiddleRow(img *cv.Mat) (bin *cv.Mat, y, first, last int, ok bool) {
	if img == nil || img.Empty() {
		return nil, 0, 0, 0, false
	}
	gray := img
	if img.Channels != 1 {
		gray = cv.CvtColor(img, cv.ColorRGB2Gray)
	}
	bin, _ = cv.Threshold(gray, 0, 255, cv.ThreshBinaryInv|cv.ThreshOtsu)
	w := bin.Cols
	y = bin.Rows / 2
	first, last = -1, -1
	for x := 0; x < w; x++ {
		if bin.Data[y*w+x] != 0 {
			if first < 0 {
				first = x
			}
			last = x
		}
	}
	if first < 0 || last <= first {
		return nil, 0, 0, 0, false
	}
	return bin, y, first, last, true
}

// recoverModules1D reads the middle scanline of img and reconstructs the module
// bit array (true = dark) by quantising every bar/space run against the
// narrowest run (one module). It requires the symbol to start and end with a
// dark bar, which every symbology in this package guarantees. It returns the
// modules and true, or nil and false on failure.
func recoverModules1D(img *cv.Mat) ([]bool, bool) {
	bin, y, first, last, ok := binarizeMiddleRow(img)
	if !ok {
		return nil, false
	}
	runs := runWidths(binRow(bin, y, first, last)) // first run is dark
	if len(runs) == 0 {
		return nil, false
	}
	unit := runs[0]
	for _, r := range runs {
		if r < unit {
			unit = r
		}
	}
	if unit <= 0 {
		return nil, false
	}
	var mods []bool
	dark := true
	for _, r := range runs {
		n := int(float64(r)/float64(unit) + 0.5)
		if n < 1 {
			n = 1
		}
		for i := 0; i < n; i++ {
			mods = append(mods, dark)
		}
		dark = !dark
	}
	return mods, true
}

// elementsNW classifies the bar/space runs of a module bit array into
// narrow/wide flags (false = narrow, true = wide), starting with the first
// (dark) bar. It is the front end for the two-width symbologies.
func elementsNW(mods []bool) []bool {
	rw := runWidths(mods)
	out := make([]bool, len(rw))
	for i, r := range rw {
		out[i] = r >= 2
	}
	return out
}

// appendElement appends a single symbology element — a bar or a space, narrow or
// wide — to a module slice, expanding a wide element to onedWide modules.
func appendElement(mods []bool, bar, wide bool) []bool {
	w := 1
	if wide {
		w = onedWide
	}
	for i := 0; i < w; i++ {
		mods = append(mods, bar)
	}
	return mods
}

// sampleFixed1D samples exactly n equally spaced modules across the dark extent
// of the middle scanline, used by the fixed-width guarded symbologies (EAN-8,
// UPC-A). It requires the symbol to end with a dark bar (the end guard does).
func sampleFixed1D(img *cv.Mat, n int) ([]bool, bool) {
	bin, y, first, last, ok := binarizeMiddleRow(img)
	if !ok {
		return nil, false
	}
	if last-first+1 < n {
		return nil, false
	}
	w := bin.Cols
	span := float64(last - first + 1)
	mod := make([]bool, n)
	for k := 0; k < n; k++ {
		x := first + int((float64(k)+0.5)*span/float64(n))
		if x >= w {
			x = w - 1
		}
		mod[k] = bin.Data[y*w+x] != 0
	}
	return mod, true
}
