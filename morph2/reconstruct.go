package morph2

import cv "github.com/malcolmston/opencv"

// ReconstructByDilation performs grey-scale morphological reconstruction of the
// mask from the marker by iterated geodesic dilation until stability. The
// marker is first clamped to be no greater than the mask. The result is the
// largest image that is bounded above by the mask and whose regional maxima are
// seeded by the marker; intuitively, it grows the marker to fill the connected
// intensity domes of the mask that it touches.
//
// It uses Vincent's fast hybrid algorithm (raster/anti-raster scans followed by
// a FIFO queue), so it is far faster than naive iteration. It panics unless
// marker and mask are single-channel and identically sized.
func ReconstructByDilation(marker, mask *cv.Mat, conn Connectivity) *cv.Mat {
	requireSameSize(marker, mask)
	_ = neighbourOffsets(conn) // validate connectivity
	rows, cols := mask.Rows, mask.Cols
	j := make([]uint8, len(mask.Data))
	for i := range j {
		j[i] = minU8(marker.Data[i], mask.Data[i])
	}
	I := mask.Data

	eightBefore := [][2]int{{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}}
	eightAfter := [][2]int{{1, 1}, {1, 0}, {1, -1}, {0, 1}}
	fourBefore := [][2]int{{-1, 0}, {0, -1}}
	fourAfter := [][2]int{{1, 0}, {0, 1}}
	before, after := eightBefore, eightAfter
	if conn == Conn4 {
		before, after = fourBefore, fourAfter
	}
	full := neighbourOffsets(conn)

	// Raster scan.
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			p := idx(y, x, cols)
			m := j[p]
			for _, o := range before {
				yy, xx := y+o[0], x+o[1]
				if yy >= 0 && yy < rows && xx >= 0 && xx < cols {
					m = maxU8(m, j[idx(yy, xx, cols)])
				}
			}
			j[p] = minU8(m, I[p])
		}
	}

	queue := make([]int, 0, len(j)/4+1)
	// Anti-raster scan.
	for y := rows - 1; y >= 0; y-- {
		for x := cols - 1; x >= 0; x-- {
			p := idx(y, x, cols)
			m := j[p]
			for _, o := range after {
				yy, xx := y+o[0], x+o[1]
				if yy >= 0 && yy < rows && xx >= 0 && xx < cols {
					m = maxU8(m, j[idx(yy, xx, cols)])
				}
			}
			j[p] = minU8(m, I[p])
			for _, o := range after {
				yy, xx := y+o[0], x+o[1]
				if yy >= 0 && yy < rows && xx >= 0 && xx < cols {
					q := idx(yy, xx, cols)
					if j[q] < j[p] && j[q] < I[q] {
						queue = append(queue, p)
						break
					}
				}
			}
		}
	}

	// FIFO propagation.
	for head := 0; head < len(queue); head++ {
		p := queue[head]
		py, px := p/cols, p%cols
		jp := j[p]
		for _, o := range full {
			yy, xx := py+o[0], px+o[1]
			if yy < 0 || yy >= rows || xx < 0 || xx >= cols {
				continue
			}
			q := idx(yy, xx, cols)
			if j[q] < jp && I[q] != j[q] {
				j[q] = minU8(jp, I[q])
				queue = append(queue, q)
			}
		}
	}

	out := cv.NewMat(rows, cols, 1)
	copy(out.Data, j)
	return out
}

// ReconstructByErosion performs grey-scale morphological reconstruction by
// erosion: the dual of [ReconstructByDilation]. The marker is clamped to be no
// smaller than the mask and shrunk by iterated geodesic erosion until
// stability. It panics unless marker and mask are single-channel and
// identically sized.
func ReconstructByErosion(marker, mask *cv.Mat, conn Connectivity) *cv.Mat {
	requireSameSize(marker, mask)
	return Complement(ReconstructByDilation(Complement(marker), Complement(mask), conn))
}

// OpenByReconstruction returns the opening by reconstruction of src: an erosion
// by e used as the marker, reconstructed under src as the mask. Unlike a plain
// opening it preserves the exact shape of the surviving components rather than
// rounding them to the structuring element.
func OpenByReconstruction(src *cv.Mat, e *Element, conn Connectivity) *cv.Mat {
	requireGray(src)
	return ReconstructByDilation(Erode(src, e), src, conn)
}

// CloseByReconstruction returns the closing by reconstruction of src: a
// dilation by e used as the marker, reconstructed by erosion under src as the
// mask. It is the dual of [OpenByReconstruction].
func CloseByReconstruction(src *cv.Mat, e *Element, conn Connectivity) *cv.Mat {
	requireGray(src)
	return ReconstructByErosion(Dilate(src, e), src, conn)
}

// FillHoles fills holes — background regions not connected to the image border —
// in a grey-scale or binary image. For grey-scale input it fills intensity
// basins; for binary input it fills enclosed background pockets. Connectivity
// refers to the background. It panics on multi-channel input.
func FillHoles(src *cv.Mat, conn Connectivity) *cv.Mat {
	requireGray(src)
	rows, cols := src.Rows, src.Cols
	// Marker: border pixels take the complemented source value, interior is 0
	// (max of the complemented mask is grown inward from the border).
	comp := Complement(src)
	marker := cv.NewMat(rows, cols, 1)
	for x := 0; x < cols; x++ {
		marker.Data[idx(0, x, cols)] = comp.Data[idx(0, x, cols)]
		marker.Data[idx(rows-1, x, cols)] = comp.Data[idx(rows-1, x, cols)]
	}
	for y := 0; y < rows; y++ {
		marker.Data[idx(y, 0, cols)] = comp.Data[idx(y, 0, cols)]
		marker.Data[idx(y, cols-1, cols)] = comp.Data[idx(y, cols-1, cols)]
	}
	rec := ReconstructByDilation(marker, comp, conn)
	return Complement(rec)
}

// ClearBorder removes connected foreground components that touch the image
// border, leaving interior components untouched. Connectivity refers to the
// foreground. It panics on multi-channel input.
func ClearBorder(src *cv.Mat, conn Connectivity) *cv.Mat {
	requireGray(src)
	rows, cols := src.Rows, src.Cols
	marker := cv.NewMat(rows, cols, 1)
	for x := 0; x < cols; x++ {
		marker.Data[idx(0, x, cols)] = src.Data[idx(0, x, cols)]
		marker.Data[idx(rows-1, x, cols)] = src.Data[idx(rows-1, x, cols)]
	}
	for y := 0; y < rows; y++ {
		marker.Data[idx(y, 0, cols)] = src.Data[idx(y, 0, cols)]
		marker.Data[idx(y, cols-1, cols)] = src.Data[idx(y, cols-1, cols)]
	}
	borderComponents := ReconstructByDilation(marker, src, conn)
	return Subtract(src, borderComponents)
}

// HMaxima suppresses all bright domes of dynamic (height) less than h by
// reconstructing src from src minus h. It removes shallow local maxima while
// preserving the position and shape of the significant ones. It panics on
// multi-channel input.
func HMaxima(src *cv.Mat, h uint8, conn Connectivity) *cv.Mat {
	requireGray(src)
	marker := newLike(src)
	for i, v := range src.Data {
		if v > h {
			marker.Data[i] = v - h
		}
	}
	return ReconstructByDilation(marker, src, conn)
}

// HMinima fills all dark basins of depth less than h by reconstructing src (by
// erosion) from src plus h. It is the dual of [HMaxima]. It panics on
// multi-channel input.
func HMinima(src *cv.Mat, h uint8, conn Connectivity) *cv.Mat {
	requireGray(src)
	marker := newLike(src)
	for i, v := range src.Data {
		s := int(v) + int(h)
		if s > 255 {
			s = 255
		}
		marker.Data[i] = uint8(s)
	}
	return ReconstructByErosion(marker, src, conn)
}

// RegionalMaxima returns a binary image marking the regional maxima of src:
// connected plateaus whose neighbours are all strictly lower. It panics on
// multi-channel input.
func RegionalMaxima(src *cv.Mat, conn Connectivity) *cv.Mat {
	requireGray(src)
	marker := newLike(src)
	for i, v := range src.Data {
		if v > 0 {
			marker.Data[i] = v - 1
		}
	}
	rec := ReconstructByDilation(marker, src, conn)
	out := newLike(src)
	for i := range src.Data {
		if src.Data[i] > rec.Data[i] {
			out.Data[i] = 255
		}
	}
	return out
}

// RegionalMinima returns a binary image marking the regional minima of src:
// connected plateaus whose neighbours are all strictly higher. It panics on
// multi-channel input.
func RegionalMinima(src *cv.Mat, conn Connectivity) *cv.Mat {
	requireGray(src)
	return RegionalMaxima(Complement(src), conn)
}

// ExtendedMaxima returns the regional maxima of the h-maxima transform of src,
// i.e. the significant bright domes of dynamic at least h, as a binary image.
// It panics on multi-channel input.
func ExtendedMaxima(src *cv.Mat, h uint8, conn Connectivity) *cv.Mat {
	return RegionalMaxima(HMaxima(src, h, conn), conn)
}

// ExtendedMinima returns the regional minima of the h-minima transform of src,
// i.e. the significant dark basins of depth at least h, as a binary image. It
// panics on multi-channel input.
func ExtendedMinima(src *cv.Mat, h uint8, conn Connectivity) *cv.Mat {
	return RegionalMinima(HMinima(src, h, conn), conn)
}

// ImposeMinima forces regional minima of value 0 at every non-zero pixel of the
// binary markerMask and raises the rest of src so that no other minima remain.
// The result is suitable as input to a watershed transform to avoid
// over-segmentation. Connectivity refers to the reconstruction. It panics
// unless src and markerMask are single-channel and identically sized.
func ImposeMinima(src, markerMask *cv.Mat, conn Connectivity) *cv.Mat {
	requireSameSize(src, markerMask)
	fm := newLike(src)   // 0 on markers, 255 elsewhere
	work := newLike(src) // pointwise min of (src+1) and fm
	for i := range src.Data {
		if markerMask.Data[i] != 0 {
			fm.Data[i] = 0
		} else {
			fm.Data[i] = 255
		}
		s := int(src.Data[i]) + 1
		if s > 255 {
			s = 255
		}
		work.Data[i] = minU8(uint8(s), fm.Data[i])
	}
	return ReconstructByErosion(fm, work, conn)
}
