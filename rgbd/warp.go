package rgbd

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// WarpFrame renders how an intensity image and its depth map would look from a
// second camera pose, the operation OpenCV calls warpFrame. Every valid source
// pixel is back-projected with intrinsics K, transformed by pose (applied as
// p' = R·p + T, mapping source-camera points into the destination camera), and
// re-projected with K into the destination image. The source intensity is
// splatted to the pixel it lands on, keeping the nearest surface via a depth
// (Z) buffer so nearer geometry occludes farther geometry.
//
// It returns the warped intensity image, the warped depth map (the destination
// Z of whatever surface won each pixel) and a boolean mask, all sized
// rows×cols. Mask entry (v,u) is true where a source point was rendered. Holes
// left by forward splatting stay zero with a false mask. It panics if image or
// depth is nil, their sizes differ, K has a zero focal length, or the output
// size is not positive.
func WarpFrame(image, depth *cv.FloatMat, k [3][3]float64, pose Pose, rows, cols int) (warpedImage, warpedDepth *cv.FloatMat, mask []bool) {
	if image == nil || depth == nil {
		panic("rgbd: WarpFrame given a nil image or depth map")
	}
	if image.Rows != depth.Rows || image.Cols != depth.Cols {
		panic("rgbd: WarpFrame image and depth sizes differ")
	}
	if rows <= 0 || cols <= 0 {
		panic("rgbd: WarpFrame requires a positive output size")
	}
	validK(k)
	fx, fy := k[0][0], k[1][1]
	cx, cy := k[0][2], k[1][2]

	warpedImage = cv.NewFloatMat(rows, cols)
	warpedDepth = cv.NewFloatMat(rows, cols)
	mask = make([]bool, rows*cols)
	for v := 0; v < depth.Rows; v++ {
		for u := 0; u < depth.Cols; u++ {
			z := depth.At(v, u)
			if z <= 0 {
				continue
			}
			p := pose.Apply(backProject(u, v, z, k))
			if p[2] <= 0 {
				continue
			}
			ou := int(math.Round(fx*p[0]/p[2] + cx))
			ov := int(math.Round(fy*p[1]/p[2] + cy))
			if ou < 0 || ou >= cols || ov < 0 || ov >= rows {
				continue
			}
			idx := ov*cols + ou
			if mask[idx] && warpedDepth.Data[idx] <= p[2] {
				continue // an existing, nearer surface already owns this pixel
			}
			warpedDepth.Data[idx] = p[2]
			warpedImage.Data[idx] = image.At(v, u)
			mask[idx] = true
		}
	}
	return warpedImage, warpedDepth, mask
}
