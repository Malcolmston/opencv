package videoproc

import (
	cv "github.com/malcolmston/opencv"
)

// AbsDiff returns the per-sample absolute difference |a-b| of two frames of
// identical dimensions and channel count. The result has the same shape as the
// inputs. It panics if the frames differ in size or channels.
func AbsDiff(a, b *cv.Mat) *cv.Mat {
	videoprocRequireSame("AbsDiff", a, b)
	out := cv.NewMat(a.Rows, a.Cols, a.Channels)
	for i := range a.Data {
		av := a.Data[i]
		bv := b.Data[i]
		if av >= bv {
			out.Data[i] = av - bv
		} else {
			out.Data[i] = bv - av
		}
	}
	return out
}

// FrameDifference returns a binary motion mask (255 = motion, 0 = static) by
// thresholding the absolute grayscale difference between prev and cur: a pixel
// is marked as motion when |cur-prev| > threshold. Multi-channel frames are
// converted to grayscale first. The frames must have identical dimensions.
func FrameDifference(prev, cur *cv.Mat, threshold uint8) *cv.Mat {
	if prev == nil || cur == nil || prev.Empty() || cur.Empty() {
		panic("videoproc: FrameDifference requires two non-empty frames")
	}
	if prev.Rows != cur.Rows || prev.Cols != cur.Cols {
		panic("videoproc: FrameDifference frame size mismatch")
	}
	gp := videoprocToGray(prev)
	gc := videoprocToGray(cur)
	out := cv.NewMat(gp.Rows, gp.Cols, 1)
	for i := range gp.Data {
		var d int
		if gc.Data[i] >= gp.Data[i] {
			d = int(gc.Data[i]) - int(gp.Data[i])
		} else {
			d = int(gp.Data[i]) - int(gc.Data[i])
		}
		if d > int(threshold) {
			out.Data[i] = 255
		}
	}
	return out
}

// ThreeFrameDifference returns a binary motion mask using the classic
// three-frame difference: a pixel is marked as motion (255) only when it changes
// between both prev→cur and cur→next by more than threshold. This suppresses the
// "ghost" trailing edges that a two-frame difference leaves behind. All three
// frames must share dimensions; multi-channel frames are converted to grayscale.
func ThreeFrameDifference(prev, cur, next *cv.Mat, threshold uint8) *cv.Mat {
	d1 := FrameDifference(prev, cur, threshold)
	d2 := FrameDifference(cur, next, threshold)
	out := cv.NewMat(d1.Rows, d1.Cols, 1)
	for i := range d1.Data {
		if d1.Data[i] == 255 && d2.Data[i] == 255 {
			out.Data[i] = 255
		}
	}
	return out
}

// Accumulate adds the intensity of frame src into the running accumulator acc
// (acc += src). src is converted to grayscale; acc must match its dimensions and
// be single-channel. This is the unweighted counterpart of [AccumulateWeighted].
func Accumulate(src *cv.Mat, acc *cv.FloatMat) {
	g := videoprocToGray(src)
	videoprocCheckAccum("Accumulate", g, acc)
	for i := range acc.Data {
		acc.Data[i] += float64(g.Data[i])
	}
}

// AccumulateSquare adds the squared intensity of frame src into acc
// (acc += src²). Combined with [Accumulate] over N frames this yields the data
// needed for a per-pixel running variance. src is converted to grayscale.
func AccumulateSquare(src *cv.Mat, acc *cv.FloatMat) {
	g := videoprocToGray(src)
	videoprocCheckAccum("AccumulateSquare", g, acc)
	for i := range acc.Data {
		v := float64(g.Data[i])
		acc.Data[i] += v * v
	}
}

// AccumulateWeighted updates the running-average accumulator acc with frame src
// using the exponential update acc = (1-alpha)*acc + alpha*src, mirroring
// cv::accumulateWeighted. A larger alpha in (0,1] adapts faster to recent
// frames. src is converted to grayscale; acc must be single-channel and match
// its dimensions. It panics if alpha is outside (0,1].
func AccumulateWeighted(src *cv.Mat, acc *cv.FloatMat, alpha float64) {
	if alpha <= 0 || alpha > 1 {
		panic("videoproc: AccumulateWeighted requires alpha in (0,1]")
	}
	g := videoprocToGray(src)
	videoprocCheckAccum("AccumulateWeighted", g, acc)
	for i := range acc.Data {
		acc.Data[i] = (1-alpha)*acc.Data[i] + alpha*float64(g.Data[i])
	}
}

// videoprocCheckAccum panics unless the single-channel gray frame g and the
// accumulator acc have matching dimensions.
func videoprocCheckAccum(fn string, g *cv.Mat, acc *cv.FloatMat) {
	if acc == nil {
		panic("videoproc: " + fn + " nil accumulator")
	}
	if acc.Rows != g.Rows || acc.Cols != g.Cols {
		panic("videoproc: " + fn + " accumulator size mismatch")
	}
}

// CountMotionPixels returns the number of non-zero pixels in a single-channel
// mask such as the output of [FrameDifference]. It panics if mask is not
// single-channel.
func CountMotionPixels(mask *cv.Mat) int {
	if mask == nil || mask.Empty() || mask.Channels != 1 {
		panic("videoproc: CountMotionPixels requires a non-empty single-channel mask")
	}
	n := 0
	for _, v := range mask.Data {
		if v != 0 {
			n++
		}
	}
	return n
}

// MotionEnergy returns the fraction of moving pixels in a single-channel mask,
// i.e. CountMotionPixels(mask) divided by the pixel count, a value in [0,1]. It
// is a convenient global measure of how much of the frame is in motion.
func MotionEnergy(mask *cv.Mat) float64 {
	if mask == nil || mask.Empty() || mask.Channels != 1 {
		panic("videoproc: MotionEnergy requires a non-empty single-channel mask")
	}
	return float64(CountMotionPixels(mask)) / float64(mask.Total())
}

// AccumulatorToMat converts a running accumulator into a viewable
// single-channel Mat by dividing every sample by count (the number of
// accumulated frames) and clamping to 0..255. It is the standard way to turn an
// [Accumulate] sum into a mean background image. It panics if count <= 0.
func AccumulatorToMat(acc *cv.FloatMat, count int) *cv.Mat {
	if acc == nil {
		panic("videoproc: AccumulatorToMat nil accumulator")
	}
	if count <= 0 {
		panic("videoproc: AccumulatorToMat requires count > 0")
	}
	out := cv.NewMat(acc.Rows, acc.Cols, 1)
	inv := 1.0 / float64(count)
	for i := range acc.Data {
		out.Data[i] = videoprocClampU8(acc.Data[i]*inv + 0.5)
	}
	return out
}
