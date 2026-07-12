package cudalegacy

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// NeedleSegment is one arrow of an optical-flow needle map: a line from Start to
// End (in pixel coordinates) whose direction and length depict the average flow
// of one grid cell.
type NeedleSegment struct {
	Start, End cv.Point
}

// CreateOpticalFlowNeedleMap is a CPU-backed mirror of OpenCV's
// cv::cuda::createOpticalFlowNeedleMap. It summarises a dense [Flow] as a sparse
// field of arrows ("needles"): the flow is averaged over gridStep×gridStep cells
// and one arrow per cell is drawn from the cell centre in the mean-flow
// direction, its length scaled by scale. The arrows are rendered onto a fresh
// three-channel [GpuMat] the same size as the flow (black background, white
// needles) and also returned as a slice of [NeedleSegment] for callers that want
// the raw vectors.
//
// gridStep falls back to 8 when non-positive; scale falls back to 1 when
// non-positive. It panics on a nil flow. The stream is a no-op.
func CreateOpticalFlowNeedleMap(flow *Flow, gridStep int, scale float64, stream *Stream) (*GpuMat, []NeedleSegment) {
	_ = stream
	if flow == nil {
		panic("cudalegacy: CreateOpticalFlowNeedleMap given a nil flow")
	}
	if gridStep <= 0 {
		gridStep = 8
	}
	if scale <= 0 {
		scale = 1
	}
	rows, cols := flow.Rows(), flow.Cols()
	canvas := cv.NewMat(rows, cols, 3)
	white := cv.NewScalar(255, 255, 255)

	var segs []NeedleSegment
	for by := 0; by < rows; by += gridStep {
		for bx := 0; bx < cols; bx += gridStep {
			var su, sv float64
			var n int
			for y := by; y < by+gridStep && y < rows; y++ {
				for x := bx; x < bx+gridStep && x < cols; x++ {
					u, v := flow.At(y, x)
					su += u
					sv += v
					n++
				}
			}
			if n == 0 {
				continue
			}
			mu := su / float64(n)
			mv := sv / float64(n)
			cx := bx + gridStep/2
			cy := by + gridStep/2
			if cx >= cols {
				cx = cols - 1
			}
			if cy >= rows {
				cy = rows - 1
			}
			ex := int(math.Round(float64(cx) + mu*scale))
			ey := int(math.Round(float64(cy) + mv*scale))
			seg := NeedleSegment{
				Start: cv.Point{X: cx, Y: cy},
				End:   cv.Point{X: ex, Y: ey},
			}
			segs = append(segs, seg)
			cv.Line(canvas, seg.Start, seg.End, white, 1)
			drawHead(canvas, seg.Start, seg.End, white)
		}
	}
	return GpuMatFromMat(canvas), segs
}

// drawHead paints a small two-pixel arrowhead at the end of a needle so its
// direction can be read. It is a light-touch marker, not a full arrow glyph.
func drawHead(canvas *cv.Mat, start, end cv.Point, color cv.Scalar) {
	dx := float64(end.X - start.X)
	dy := float64(end.Y - start.Y)
	length := math.Hypot(dx, dy)
	if length < 1e-9 {
		return
	}
	ux := dx / length
	uy := dy / length
	// Two barbs at +/-135 degrees, length 2.
	const barb = 2.0
	for _, ang := range []float64{math.Pi * 3 / 4, -math.Pi * 3 / 4} {
		ca := math.Cos(ang)
		sa := math.Sin(ang)
		bx := ux*ca - uy*sa
		by := ux*sa + uy*ca
		p := cv.Point{
			X: int(math.Round(float64(end.X) + bx*barb)),
			Y: int(math.Round(float64(end.Y) + by*barb)),
		}
		cv.Line(canvas, end, p, color, 1)
	}
}
