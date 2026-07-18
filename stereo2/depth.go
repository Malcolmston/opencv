package stereo2

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
)

// Camera holds the pinhole intrinsics needed to back-project image pixels into
// 3-D: the focal lengths Fx, Fy (in pixels) and the principal point (Cx, Cy).
type Camera struct {
	// Fx is the horizontal focal length in pixels.
	Fx float64
	// Fy is the vertical focal length in pixels.
	Fy float64
	// Cx is the principal-point column.
	Cx float64
	// Cy is the principal-point row.
	Cy float64
}

// NewCamera returns a Camera with the given focal lengths and principal point.
func NewCamera(fx, fy, cx, cy float64) Camera {
	return Camera{Fx: fx, Fy: fy, Cx: cx, Cy: cy}
}

// DepthFromDisparity converts a disparity map to a metric depth map using the
// standard relation Z = focalLength * baseline / d, valid for a rectified pair
// with the given focal length (pixels) and baseline (world units). Pixels with
// an invalid or non-positive disparity become [InvalidDepth]. It panics if
// focalLength or baseline is not positive.
func DepthFromDisparity(disp *DisparityMap, focalLength, baseline float64) *DepthMap {
	if focalLength <= 0 || baseline <= 0 {
		panic(fmt.Sprintf("stereo2: DepthFromDisparity requires positive focalLength and baseline, got %g, %g", focalLength, baseline))
	}
	out := NewDepthMap(disp.Rows, disp.Cols)
	fb := focalLength * baseline
	for i, d := range disp.Data {
		if d <= 0 {
			continue
		}
		out.Data[i] = float32(fb / float64(d))
	}
	return out
}

// DisparityFromDepth is the inverse of [DepthFromDisparity]: it converts a metric
// depth map back to a disparity map with d = focalLength * baseline / Z. Pixels
// with an invalid or non-positive depth become [InvalidDisparity]. It panics if
// focalLength or baseline is not positive.
func DisparityFromDepth(depth *DepthMap, focalLength, baseline float64) *DisparityMap {
	if focalLength <= 0 || baseline <= 0 {
		panic(fmt.Sprintf("stereo2: DisparityFromDepth requires positive focalLength and baseline, got %g, %g", focalLength, baseline))
	}
	out := NewDisparityMap(depth.Rows, depth.Cols)
	fb := focalLength * baseline
	for i, z := range depth.Data {
		if z <= 0 {
			continue
		}
		out.Data[i] = float32(fb / float64(z))
	}
	return out
}

// PointCloud is an unordered set of reconstructed 3-D points, optionally carrying
// a per-point RGB colour sampled from a reference image.
type PointCloud struct {
	// Points holds the 3-D coordinates.
	Points []Point3D
	// Colors, when non-nil, holds one RGB triple per point in Points.
	Colors [][3]uint8
}

// Len returns the number of points in the cloud.
func (pc *PointCloud) Len() int { return len(pc.Points) }

// PointCloudFromDepth back-projects a depth map into a metric point cloud using
// the camera intrinsics: for pixel (y,x) with depth Z the point is
// ((x-Cx)*Z/Fx, (y-Cy)*Z/Fy, Z). When color is non-nil (and matches the depth
// map size) each point is tagged with that image's RGB value. Invalid pixels are
// skipped. It panics if color is non-nil but a different size.
func PointCloudFromDepth(depth *DepthMap, cam Camera, color *cv.Mat) *PointCloud {
	if color != nil && (color.Rows != depth.Rows || color.Cols != depth.Cols) {
		panic("stereo2: PointCloudFromDepth color image size mismatch")
	}
	pc := &PointCloud{}
	if color != nil {
		pc.Colors = [][3]uint8{}
	}
	for y := 0; y < depth.Rows; y++ {
		for x := 0; x < depth.Cols; x++ {
			z := float64(depth.Data[y*depth.Cols+x])
			if z <= 0 {
				continue
			}
			p := Point3D{
				X: (float64(x) - cam.Cx) * z / cam.Fx,
				Y: (float64(y) - cam.Cy) * z / cam.Fy,
				Z: z,
			}
			pc.Points = append(pc.Points, p)
			if color != nil {
				pc.Colors = append(pc.Colors, sampleColor(color, y, x))
			}
		}
	}
	return pc
}

// PointCloudFromDisparity converts a disparity map to depth (Fx as focal length
// and the given baseline) and then back-projects it with [PointCloudFromDepth].
// It panics if baseline is not positive.
func PointCloudFromDisparity(disp *DisparityMap, cam Camera, baseline float64, color *cv.Mat) *PointCloud {
	depth := DepthFromDisparity(disp, cam.Fx, baseline)
	return PointCloudFromDepth(depth, cam, color)
}

// QMatrix builds the 4x4 disparity-to-depth reprojection matrix Q for a rectified
// stereo pair with principal point (cx, cy), focal length f (pixels) and baseline
// (world units), assuming both cameras share the same principal point. It maps a
// homogeneous pixel-disparity vector [x y d 1]^T to a homogeneous 3-D point, as
// consumed by [ReprojectImageTo3D].
func QMatrix(cx, cy, f, baseline float64) [4][4]float64 {
	return [4][4]float64{
		{1, 0, 0, -cx},
		{0, 1, 0, -cy},
		{0, 0, 0, f},
		{0, 0, 1 / baseline, 0},
	}
}

// ReprojectImageTo3D maps every valid disparity to a 3-D point using the 4x4
// reprojection matrix Q (see [QMatrix]), the standard cv::reprojectImageTo3D
// operation. For pixel (y,x) with disparity d it forms [X Y Z W]^T = Q·[x y d 1]^T
// and emits (X/W, Y/W, Z/W); points with a non-positive homogeneous divisor are
// skipped. When color is non-nil (and matches the disparity size) points are
// tagged with its RGB values. It panics on a color size mismatch.
func ReprojectImageTo3D(disp *DisparityMap, Q [4][4]float64, color *cv.Mat) *PointCloud {
	if color != nil && (color.Rows != disp.Rows || color.Cols != disp.Cols) {
		panic("stereo2: ReprojectImageTo3D color image size mismatch")
	}
	pc := &PointCloud{}
	if color != nil {
		pc.Colors = [][3]uint8{}
	}
	for y := 0; y < disp.Rows; y++ {
		for x := 0; x < disp.Cols; x++ {
			d := float64(disp.Data[y*disp.Cols+x])
			if d <= 0 {
				continue
			}
			vx := float64(x)
			vy := float64(y)
			X := Q[0][0]*vx + Q[0][1]*vy + Q[0][2]*d + Q[0][3]
			Y := Q[1][0]*vx + Q[1][1]*vy + Q[1][2]*d + Q[1][3]
			Z := Q[2][0]*vx + Q[2][1]*vy + Q[2][2]*d + Q[2][3]
			W := Q[3][0]*vx + Q[3][1]*vy + Q[3][2]*d + Q[3][3]
			if W <= 0 {
				continue
			}
			pc.Points = append(pc.Points, Point3D{X / W, Y / W, Z / W})
			if color != nil {
				pc.Colors = append(pc.Colors, sampleColor(color, y, x))
			}
		}
	}
	return pc
}

// sampleColor reads pixel (y,x) of img as an RGB triple, replicating a single
// channel across all three when the image is grayscale.
func sampleColor(img *cv.Mat, y, x int) [3]uint8 {
	if img.Channels == 1 {
		g := img.At(y, x, 0)
		return [3]uint8{g, g, g}
	}
	return [3]uint8{img.At(y, x, 0), img.At(y, x, 1), img.At(y, x, 2)}
}
