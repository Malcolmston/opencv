package barcode

import cv "github.com/malcolmston/opencv"

// This file exposes QR grid sampling: turning a rendered QR image into the
// module bit matrix that the metadata and data decoders consume. It also
// provides small pure helpers describing a version's module counts. The sampler
// binarises the image, takes the bounding box of the dark region (a clean QR
// render touches all four edges with its finder patterns, so the box is exactly
// the symbol area excluding the quiet zone), and samples the centre of each of
// the version*version cells.

// QRModuleCount returns the number of modules along one side of a QR symbol of
// the given version (identical to [QRSizeForVersion]); it returns 0 for
// versions outside 1-40.
func QRModuleCount(version int) int {
	return QRSizeForVersion(version)
}

// QRRawDataModuleCount returns the number of data-carrying modules of a QR
// symbol of the given version: the total module count minus the function
// patterns (finders, separators, timing, alignment, format and, for versions
// 7+, version information). It returns 0 for versions outside 1-40.
func QRRawDataModuleCount(version int) int {
	if version < 1 || version > 40 {
		return 0
	}
	return numRawModulesEx(version)
}

// SampleQRModules samples a rendered QR image into its version*version module
// matrix, where a true element denotes a dark module. The image may be
// grayscale or colour and may carry a quiet zone; it is Otsu-binarised and the
// dark bounding box is divided into an evenly spaced grid. It returns the
// module matrix and true, or nil and false when the version is invalid or the
// image contains no dark pixels.
func SampleQRModules(img *cv.Mat, version int) ([][]bool, bool) {
	size := QRSizeForVersion(version)
	if size == 0 || img == nil || img.Empty() {
		return nil, false
	}
	dark := toDarkGrid(img)
	h := len(dark)
	if h == 0 {
		return nil, false
	}
	w := len(dark[0])
	minX, minY, maxX, maxY := w, h, -1, -1
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			if dark[y][x] {
				if x < minX {
					minX = x
				}
				if x > maxX {
					maxX = x
				}
				if y < minY {
					minY = y
				}
				if y > maxY {
					maxY = y
				}
			}
		}
	}
	if maxX < minX || maxY < minY {
		return nil, false
	}
	spanX := float64(maxX-minX+1) / float64(size)
	spanY := float64(maxY-minY+1) / float64(size)
	grid := make([][]bool, size)
	for r := 0; r < size; r++ {
		row := make([]bool, size)
		y := minY + int((float64(r)+0.5)*spanY)
		if y >= h {
			y = h - 1
		}
		for c := 0; c < size; c++ {
			x := minX + int((float64(c)+0.5)*spanX)
			if x >= w {
				x = w - 1
			}
			row[c] = dark[y][x]
		}
		grid[r] = row
	}
	return grid, true
}

// CountDarkModules returns the number of true (dark) entries in a module
// matrix.
func CountDarkModules(grid [][]bool) int {
	n := 0
	for _, row := range grid {
		for _, v := range row {
			if v {
				n++
			}
		}
	}
	return n
}

// DarkModuleRatio returns the fraction of dark modules in a module matrix, a
// value in [0,1]. A well-formed QR symbol has a ratio near 0.5; a strong
// deviation is a useful signal that sampling or binarisation went wrong. It
// returns 0 for an empty matrix.
func DarkModuleRatio(grid [][]bool) float64 {
	total := 0
	for _, row := range grid {
		total += len(row)
	}
	if total == 0 {
		return 0
	}
	return float64(CountDarkModules(grid)) / float64(total)
}
