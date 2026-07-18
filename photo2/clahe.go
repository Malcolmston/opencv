package photo2

import (
	cv "github.com/malcolmston/opencv"
)

// CLAHE performs Contrast-Limited Adaptive Histogram Equalisation. The image is
// divided into a tiles x tiles grid; a clipped, redistributed histogram is
// equalised within each tile and the per-tile mappings are bilinearly
// interpolated across the image, avoiding the blocky artefacts of plain tiled
// equalisation. clipLimit bounds the histogram slope (values around 2–4 are
// typical; <=0 disables clipping). For colour input the mapping is applied to
// luminance with colour ratios preserved.
func CLAHE(img *cv.Mat, clipLimit float64, tiles int) *cv.Mat {
	photo2RequireImage(img, "CLAHE")
	if tiles < 1 {
		tiles = 8
	}
	if img.Channels == 1 {
		mapped := photo2CLAHEPlane(img.Data, img.Rows, img.Cols, clipLimit, tiles)
		out := cv.NewMat(img.Rows, img.Cols, 1)
		copy(out.Data, mapped)
		return out
	}
	photo2RequireRGB(img, "CLAHE")
	total := img.Rows * img.Cols
	lum := make([]uint8, total)
	for i := 0; i < total; i++ {
		r := float64(img.Data[i*3+0])
		g := float64(img.Data[i*3+1])
		b := float64(img.Data[i*3+2])
		lum[i] = photo2Clamp8(photo2Luma(r, g, b))
	}
	mapped := photo2CLAHEPlane(lum, img.Rows, img.Cols, clipLimit, tiles)
	out := cv.NewMat(img.Rows, img.Cols, 3)
	for i := 0; i < total; i++ {
		oldY := float64(lum[i])
		newY := float64(mapped[i])
		ratio := 1.0
		if oldY > 0 {
			ratio = newY / oldY
		}
		for c := 0; c < 3; c++ {
			out.Data[i*3+c] = photo2Clamp8(float64(img.Data[i*3+c]) * ratio)
		}
	}
	return out
}

// photo2CLAHEPlane runs CLAHE on a single 8-bit plane and returns the mapped
// samples.
func photo2CLAHEPlane(data []uint8, rows, cols int, clipLimit float64, tiles int) []uint8 {
	tw := (cols + tiles - 1) / tiles
	th := (rows + tiles - 1) / tiles
	// Build a LUT per tile.
	luts := make([][256]uint8, tiles*tiles)
	for ty := 0; ty < tiles; ty++ {
		for tx := 0; tx < tiles; tx++ {
			x0 := tx * tw
			y0 := ty * th
			x1 := x0 + tw
			y1 := y0 + th
			if x1 > cols {
				x1 = cols
			}
			if y1 > rows {
				y1 = rows
			}
			var hist [256]int
			count := 0
			for y := y0; y < y1; y++ {
				for x := x0; x < x1; x++ {
					hist[data[y*cols+x]]++
					count++
				}
			}
			if count == 0 {
				count = 1
			}
			if clipLimit > 0 {
				limit := int(clipLimit * float64(count) / 256)
				if limit < 1 {
					limit = 1
				}
				excess := 0
				for v := 0; v < 256; v++ {
					if hist[v] > limit {
						excess += hist[v] - limit
						hist[v] = limit
					}
				}
				// Redistribute the clipped excess uniformly.
				inc := excess / 256
				rem := excess % 256
				for v := 0; v < 256; v++ {
					hist[v] += inc
				}
				for v := 0; v < rem; v++ {
					hist[v]++
				}
			}
			// CDF -> LUT.
			cdf := 0
			for v := 0; v < 256; v++ {
				cdf += hist[v]
				luts[ty*tiles+tx][v] = photo2Clamp8(float64(cdf) / float64(count) * 255)
			}
		}
	}
	// Bilinearly interpolate tile LUTs across the image using tile centres.
	out := make([]uint8, rows*cols)
	for y := 0; y < rows; y++ {
		fy := (float64(y) - float64(th)/2) / float64(th)
		ty0 := int(fy)
		if ty0 < 0 {
			ty0 = 0
		}
		ty1 := ty0 + 1
		if ty1 > tiles-1 {
			ty1 = tiles - 1
		}
		if ty0 > tiles-1 {
			ty0 = tiles - 1
		}
		wy := fy - float64(ty0)
		if wy < 0 {
			wy = 0
		}
		if wy > 1 {
			wy = 1
		}
		for x := 0; x < cols; x++ {
			fx := (float64(x) - float64(tw)/2) / float64(tw)
			tx0 := int(fx)
			if tx0 < 0 {
				tx0 = 0
			}
			tx1 := tx0 + 1
			if tx1 > tiles-1 {
				tx1 = tiles - 1
			}
			if tx0 > tiles-1 {
				tx0 = tiles - 1
			}
			wx := fx - float64(tx0)
			if wx < 0 {
				wx = 0
			}
			if wx > 1 {
				wx = 1
			}
			v := data[y*cols+x]
			a := float64(luts[ty0*tiles+tx0][v])
			b := float64(luts[ty0*tiles+tx1][v])
			c := float64(luts[ty1*tiles+tx0][v])
			d := float64(luts[ty1*tiles+tx1][v])
			top := a*(1-wx) + b*wx
			bot := c*(1-wx) + d*wx
			out[y*cols+x] = photo2Clamp8(top*(1-wy) + bot*wy)
		}
	}
	return out
}
