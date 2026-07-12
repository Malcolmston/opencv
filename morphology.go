package cv

import "fmt"

// MorphShape selects the shape of a structuring element built by
// [GetStructuringElement].
type MorphShape int

const (
	// MorphRect is a filled rectangle: every element is set.
	MorphRect MorphShape = iota
	// MorphCross is a plus/cross: only the centre row and column are set.
	MorphCross
	// MorphEllipse is a filled ellipse inscribed in the element rectangle.
	MorphEllipse
)

// GetStructuringElement returns a structuring element (kernel) of the given
// shape and size as a single-channel Mat whose set elements are 1 and unset
// elements are 0. rows and cols must be positive and odd for a well-defined
// centre anchor.
func GetStructuringElement(shape MorphShape, rows, cols int) *Mat {
	if rows <= 0 || cols <= 0 {
		panic("cv: GetStructuringElement requires positive size")
	}
	k := NewMat(rows, cols, 1)
	cy := rows / 2
	cx := cols / 2
	switch shape {
	case MorphRect:
		for i := range k.Data {
			k.Data[i] = 1
		}
	case MorphCross:
		for y := 0; y < rows; y++ {
			for x := 0; x < cols; x++ {
				if x == cx || y == cy {
					k.Data[y*cols+x] = 1
				}
			}
		}
	case MorphEllipse:
		ry := float64(cy)
		rx := float64(cx)
		if ry == 0 {
			ry = 1
		}
		if rx == 0 {
			rx = 1
		}
		for y := 0; y < rows; y++ {
			for x := 0; x < cols; x++ {
				dy := (float64(y) - float64(cy)) / ry
				dx := (float64(x) - float64(cx)) / rx
				if dx*dx+dy*dy <= 1.0 {
					k.Data[y*cols+x] = 1
				}
			}
		}
	default:
		panic(fmt.Sprintf("cv: GetStructuringElement unknown shape %d", shape))
	}
	return k
}

// morph applies a min (erosion) or max (dilation) rank filter over the set
// elements of kernel. iterations repeats the operation.
func morph(src, kernel *Mat, dilate bool, iterations int) *Mat {
	if kernel.Channels != 1 {
		panic("cv: morphology kernel must be single-channel")
	}
	if iterations < 1 {
		iterations = 1
	}
	ay := kernel.Rows / 2
	ax := kernel.Cols / 2
	cur := src
	for it := 0; it < iterations; it++ {
		dst := NewMat(cur.Rows, cur.Cols, cur.Channels)
		for y := 0; y < cur.Rows; y++ {
			for x := 0; x < cur.Cols; x++ {
				for c := 0; c < cur.Channels; c++ {
					var acc uint8
					if dilate {
						acc = 0
					} else {
						acc = 255
					}
					for ky := 0; ky < kernel.Rows; ky++ {
						for kx := 0; kx < kernel.Cols; kx++ {
							if kernel.Data[ky*kernel.Cols+kx] == 0 {
								continue
							}
							v := cur.atReplicate(y+ky-ay, x+kx-ax, c)
							if dilate {
								if v > acc {
									acc = v
								}
							} else {
								if v < acc {
									acc = v
								}
							}
						}
					}
					dst.Data[dst.index(y, x)+c] = acc
				}
			}
		}
		cur = dst
	}
	return cur
}

// Erode shrinks bright regions by replacing each sample with the minimum over
// the structuring element's footprint. iterations repeats the operation (a
// value < 1 is treated as 1).
func Erode(src, kernel *Mat, iterations int) *Mat {
	return morph(src, kernel, false, iterations)
}

// Dilate grows bright regions by replacing each sample with the maximum over
// the structuring element's footprint. iterations repeats the operation.
func Dilate(src, kernel *Mat, iterations int) *Mat {
	return morph(src, kernel, true, iterations)
}

// MorphType selects the compound operation performed by [MorphologyEx].
type MorphType int

const (
	// MorphErode is a plain erosion.
	MorphErode MorphType = iota
	// MorphDilate is a plain dilation.
	MorphDilate
	// MorphOpen is erosion followed by dilation; it removes small bright specks.
	MorphOpen
	// MorphClose is dilation followed by erosion; it fills small dark holes.
	MorphClose
	// MorphGradient is dilation minus erosion; it highlights edges.
	MorphGradient
	// MorphTophat is source minus its opening; it isolates small bright details.
	MorphTophat
	// MorphBlackhat is the closing minus the source; it isolates small dark
	// details.
	MorphBlackhat
)

// MorphologyEx performs a compound morphological operation selected by op over
// the given structuring element. Subtractive operations saturate at zero.
func MorphologyEx(src, kernel *Mat, op MorphType, iterations int) *Mat {
	switch op {
	case MorphErode:
		return Erode(src, kernel, iterations)
	case MorphDilate:
		return Dilate(src, kernel, iterations)
	case MorphOpen:
		return Dilate(Erode(src, kernel, iterations), kernel, iterations)
	case MorphClose:
		return Erode(Dilate(src, kernel, iterations), kernel, iterations)
	case MorphGradient:
		return subtractMat(Dilate(src, kernel, iterations), Erode(src, kernel, iterations))
	case MorphTophat:
		return subtractMat(src, Dilate(Erode(src, kernel, iterations), kernel, iterations))
	case MorphBlackhat:
		return subtractMat(Erode(Dilate(src, kernel, iterations), kernel, iterations), src)
	default:
		panic(fmt.Sprintf("cv: MorphologyEx unknown op %d", op))
	}
}

// subtractMat computes a - b per sample, saturating at zero.
func subtractMat(a, b *Mat) *Mat {
	if a.Rows != b.Rows || a.Cols != b.Cols || a.Channels != b.Channels {
		panic("cv: subtract shape mismatch")
	}
	out := NewMat(a.Rows, a.Cols, a.Channels)
	for i := range a.Data {
		if a.Data[i] > b.Data[i] {
			out.Data[i] = a.Data[i] - b.Data[i]
		}
	}
	return out
}
