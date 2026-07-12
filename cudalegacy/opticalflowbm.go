package cudalegacy

// CalcOpticalFlowBM is a CPU-backed mirror of OpenCV's cv::cuda::calcOpticalFlowBM
// — dense optical flow by exhaustive block matching. The previous frame is
// tiled into blockSize×blockSize blocks stepped by shiftSize; for each block the
// same-sized patch in curr that minimises the sum of squared differences over
// the search window [-maxRange, maxRange]² is found, and its integer
// displacement is written to every pixel of the block. The result is a dense,
// per-pixel [Flow].
//
// Both frames must be non-empty and identically shaped. blockSize and shiftSize
// fall back to 8 and 1 when non-positive; maxRange falls back to blockSize.
// Frames are reduced to intensity before matching. It panics on nil, empty or
// mismatched frames. The stream is a no-op.
func CalcOpticalFlowBM(prev, curr *GpuMat, blockSize, shiftSize, maxRange int, stream *Stream) *Flow {
	_ = stream
	pm := requireMat(prev, "CalcOpticalFlowBM")
	cm := requireMat(curr, "CalcOpticalFlowBM")
	if pm.Rows != cm.Rows || pm.Cols != cm.Cols {
		panic("cudalegacy: CalcOpticalFlowBM frames differ in size")
	}
	if blockSize <= 0 {
		blockSize = 8
	}
	if shiftSize <= 0 {
		shiftSize = 1
	}
	if maxRange <= 0 {
		maxRange = blockSize
	}
	rows, cols := pm.Rows, pm.Cols
	p := grayPlane(pm)
	c := grayPlane(cm)
	flow := NewFlow(rows, cols)

	blockSSD := func(by, bx, dy, dx int) float64 {
		sum := 0.0
		for y := 0; y < blockSize; y++ {
			py := by + y
			if py >= rows {
				break
			}
			cy := py + dy
			if cy < 0 || cy >= rows {
				return inf
			}
			for x := 0; x < blockSize; x++ {
				px := bx + x
				if px >= cols {
					break
				}
				cx := px + dx
				if cx < 0 || cx >= cols {
					return inf
				}
				d := p.Data[py*cols+px] - c.Data[cy*cols+cx]
				sum += d * d
			}
		}
		return sum
	}

	for by := 0; by < rows; by += shiftSize {
		for bx := 0; bx < cols; bx += shiftSize {
			bestSSD := inf
			bestDX, bestDY := 0, 0
			for dy := -maxRange; dy <= maxRange; dy++ {
				for dx := -maxRange; dx <= maxRange; dx++ {
					s := blockSSD(by, bx, dy, dx)
					if s < bestSSD {
						bestSSD = s
						bestDX, bestDY = dx, dy
					}
				}
			}
			// Assign the block displacement to the covered pixels.
			for y := 0; y < shiftSize && by+y < rows; y++ {
				for x := 0; x < shiftSize && bx+x < cols; x++ {
					flow.set(by+y, bx+x, float64(bestDX), float64(bestDY))
				}
			}
		}
	}
	return flow
}

// inf is a large finite sentinel used as an initial "no match" SSD.
const inf = 1e308
