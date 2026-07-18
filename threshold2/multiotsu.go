package threshold2

import (
	"errors"
	"math"

	cv "github.com/malcolmston/opencv"
)

// MultiOtsu returns classes-1 ascending thresholds that split the intensity
// range into the requested number of classes by maximising the total
// between-class variance (the multi-level generalisation of Otsu's method).
// classes must be at least 2 and at most 6.
func MultiOtsu(src *cv.Mat, classes int) ([]int, error) {
	if classes < 2 || classes > 6 {
		return nil, errors.New("threshold2: MultiOtsu classes must be in [2,6]")
	}
	bins, total, err := threshold2binsOf(src)
	if err != nil {
		return nil, err
	}
	if total == 0 {
		return make([]int, classes-1), nil
	}
	tot := float64(total)
	// Cumulative probability and cumulative i*probability, 1-indexed so that
	// index 0 represents "nothing accumulated yet".
	var P, S [257]float64
	for i := 0; i < 256; i++ {
		p := float64(bins[i]) / tot
		P[i+1] = P[i] + p
		S[i+1] = S[i] + float64(i)*p
	}
	// classMoment returns w*mu^2 for the inclusive bin range [a,b].
	classMoment := func(a, b int) float64 {
		w := P[b+1] - P[a]
		if w <= 0 {
			return 0
		}
		s := S[b+1] - S[a]
		return s * s / w
	}

	k := classes - 1
	thresholds := make([]int, k)
	best := make([]int, k)
	bestVar := math.Inf(-1)

	var recurse func(depth, start int, acc float64)
	recurse = func(depth, start int, acc float64) {
		if depth == k {
			v := acc + classMoment(thresholds[k-1]+1, 255)
			if v > bestVar {
				bestVar = v
				copy(best, thresholds)
			}
			return
		}
		prevStart := 0
		if depth > 0 {
			prevStart = thresholds[depth-1] + 1
		}
		for t := start; t < 256-(k-depth-1); t++ {
			thresholds[depth] = t
			recurse(depth+1, t+1, acc+classMoment(prevStart, t))
		}
	}
	recurse(0, 1, 0)
	return best, nil
}

// MultiOtsuQuantize applies [MultiOtsu] and returns a single-channel image in
// which every pixel is mapped to the mid-grey of its class, together with the
// thresholds used.
func MultiOtsuQuantize(src *cv.Mat, classes int) (*cv.Mat, []int, error) {
	ts, err := MultiOtsu(src, classes)
	if err != nil {
		return nil, nil, err
	}
	gray, rows, cols, err := threshold2gray(src)
	if err != nil {
		return nil, nil, err
	}
	return threshold2quantize(gray, rows, cols, ts), ts, nil
}

// threshold23x3mean returns the 3x3 neighbourhood mean (rounded) of each pixel,
// using edge replication at the borders.
func threshold23x3mean(gray []uint8, rows, cols int) []uint8 {
	out := make([]uint8, rows*cols)
	at := func(y, x int) int {
		if y < 0 {
			y = 0
		} else if y >= rows {
			y = rows - 1
		}
		if x < 0 {
			x = 0
		} else if x >= cols {
			x = cols - 1
		}
		return int(gray[y*cols+x])
	}
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			sum := 0
			for dy := -1; dy <= 1; dy++ {
				for dx := -1; dx <= 1; dx++ {
					sum += at(y+dy, x+dx)
				}
			}
			out[y*cols+x] = uint8((sum + 4) / 9)
		}
	}
	return out
}

// Otsu2DThreshold returns the two-dimensional Otsu threshold pair (s, t): s is
// the threshold on the pixel grey level and t the threshold on the 3x3
// neighbourhood-mean grey level. The pair maximises the trace of the
// between-class scatter of the joint (intensity, local-mean) histogram, which
// makes it more robust to noise than the 1-D method.
func Otsu2DThreshold(src *cv.Mat) (s, t int, err error) {
	gray, rows, cols, err := threshold2gray(src)
	if err != nil {
		return 0, 0, err
	}
	means := threshold23x3mean(gray, rows, cols)
	n := float64(len(gray))

	// Joint histogram and its cumulative moments, 1-indexed.
	var P, Si, Sj [257][257]float64
	var jh [256][256]float64
	for p := range gray {
		jh[gray[p]][means[p]]++
	}
	for i := 1; i <= 256; i++ {
		for j := 1; j <= 256; j++ {
			pij := jh[i-1][j-1] / n
			P[i][j] = pij + P[i-1][j] + P[i][j-1] - P[i-1][j-1]
			Si[i][j] = float64(i-1)*pij + Si[i-1][j] + Si[i][j-1] - Si[i-1][j-1]
			Sj[i][j] = float64(j-1)*pij + Sj[i-1][j] + Sj[i][j-1] - Sj[i-1][j-1]
		}
	}
	muTi := Si[256][256]
	muTj := Sj[256][256]

	bestS, bestT := 0, 0
	bestVar := math.Inf(-1)
	for si := 0; si < 256; si++ {
		for tj := 0; tj < 256; tj++ {
			// Background: i in [0,si], j in [0,tj].
			w0 := P[si+1][tj+1]
			// Foreground: i in [si+1,255], j in [tj+1,255].
			w1 := P[256][256] - P[si+1][256] - P[256][tj+1] + P[si+1][tj+1]
			if w0 <= 0 || w1 <= 0 {
				continue
			}
			mu0i := Si[si+1][tj+1] / w0
			mu0j := Sj[si+1][tj+1] / w0
			s1i := Si[256][256] - Si[si+1][256] - Si[256][tj+1] + Si[si+1][tj+1]
			s1j := Sj[256][256] - Sj[si+1][256] - Sj[256][tj+1] + Sj[si+1][tj+1]
			mu1i := s1i / w1
			mu1j := s1j / w1
			v := w0*((mu0i-muTi)*(mu0i-muTi)+(mu0j-muTj)*(mu0j-muTj)) +
				w1*((mu1i-muTi)*(mu1i-muTi)+(mu1j-muTj)*(mu1j-muTj))
			if v > bestVar {
				bestVar = v
				bestS = si
				bestT = tj
			}
		}
	}
	return bestS, bestT, nil
}

// Otsu2D binarizes src with the two-dimensional Otsu method. A pixel is
// foreground when both its grey level and its 3x3 neighbourhood mean fall on
// the foreground side of the [Otsu2DThreshold] pair (greater than the pair for
// [ObjectBright], at most the pair for [ObjectDark]). It returns the mask and
// the (s, t) pair used.
func Otsu2D(src *cv.Mat, p Polarity) (*cv.Mat, int, int, error) {
	s, t, err := Otsu2DThreshold(src)
	if err != nil {
		return nil, 0, 0, err
	}
	gray, rows, cols, _ := threshold2gray(src)
	means := threshold23x3mean(gray, rows, cols)
	dst := cv.NewMat(rows, cols, 1)
	for i := range gray {
		var fg bool
		if p == ObjectDark {
			fg = int(gray[i]) <= s && int(means[i]) <= t
		} else {
			fg = int(gray[i]) > s && int(means[i]) > t
		}
		if fg {
			dst.Data[i] = 255
		}
	}
	return dst, s, t, nil
}
