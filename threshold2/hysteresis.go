package threshold2

import (
	"errors"

	cv "github.com/malcolmston/opencv"
)

// Hysteresis performs double-threshold (hysteresis) binarization, the selection
// rule used by the Canny edge detector. Pixels at or above high are seed
// foreground; pixels below low are background; pixels in between become
// foreground only if they are connected (8-connectivity) to a seed through a
// chain of in-between-or-above pixels. The result is a single-channel mask with
// foreground set to 255. It requires 0 <= low <= high <= 255.
func Hysteresis(src *cv.Mat, low, high int) (*cv.Mat, error) {
	if low > high {
		return nil, errors.New("threshold2: Hysteresis requires low <= high")
	}
	gray, rows, cols, err := threshold2gray(src)
	if err != nil {
		return nil, err
	}
	dst := cv.NewMat(rows, cols, 1)
	// state: 0 = background/undecided, 1 = weak (>=low), 2 = strong (>=high).
	state := make([]uint8, rows*cols)
	stack := make([]int, 0, rows*cols/4+1)
	for i, v := range gray {
		switch {
		case int(v) >= high:
			state[i] = 2
			dst.Data[i] = 255
			stack = append(stack, i)
		case int(v) >= low:
			state[i] = 1
		}
	}
	for len(stack) > 0 {
		i := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		y := i / cols
		x := i % cols
		for dy := -1; dy <= 1; dy++ {
			ny := y + dy
			if ny < 0 || ny >= rows {
				continue
			}
			for dx := -1; dx <= 1; dx++ {
				nx := x + dx
				if nx < 0 || nx >= cols || (dx == 0 && dy == 0) {
					continue
				}
				j := ny*cols + nx
				if state[j] == 1 {
					state[j] = 2
					dst.Data[j] = 255
					stack = append(stack, j)
				}
			}
		}
	}
	return dst, nil
}
