package rgbd

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// TSDFVolume is a lightweight truncated signed distance function volume, the
// core data structure of KinectFusion-style dense reconstruction. Space is
// discretised into a regular grid of cubic voxels; each voxel stores a truncated
// signed distance to the nearest observed surface (negative behind the surface,
// positive in front) and an accumulated confidence weight. Fusing many depth
// frames with [TSDFVolume.Integrate] averages out sensor noise, and the surface
// can be recovered with [TSDFVolume.Raycast] or [TSDFVolume.FetchPointCloud].
//
// The zero value is not usable; build one with [NewTSDFVolume].
type TSDFVolume struct {
	dims       [3]int     // voxel counts along X, Y, Z
	voxelSize  float64    // side length of a voxel, in metres
	origin     [3]float64 // world position of voxel (0,0,0)'s corner
	truncation float64    // truncation distance for the signed distance
	tsdf       []float64  // per-voxel truncated signed distance (in [-1,1])
	weight     []float64  // per-voxel accumulated weight
}

// NewTSDFVolume allocates an empty volume of dims = {nx, ny, nz} voxels of side
// voxelSize (metres), whose (0,0,0) corner sits at origin in world coordinates.
// truncation is the signed-distance truncation band (metres); a value ≤ 0
// selects a default of five voxels. It panics if any dimension or the voxel size
// is not positive.
func NewTSDFVolume(dims [3]int, voxelSize float64, origin [3]float64, truncation float64) *TSDFVolume {
	if dims[0] <= 0 || dims[1] <= 0 || dims[2] <= 0 {
		panic("rgbd: NewTSDFVolume requires positive dimensions")
	}
	if voxelSize <= 0 {
		panic("rgbd: NewTSDFVolume requires a positive voxel size")
	}
	if truncation <= 0 {
		truncation = 5 * voxelSize
	}
	n := dims[0] * dims[1] * dims[2]
	return &TSDFVolume{
		dims:       dims,
		voxelSize:  voxelSize,
		origin:     origin,
		truncation: truncation,
		tsdf:       make([]float64, n),
		weight:     make([]float64, n),
	}
}

// idx maps voxel coordinates to a flat index (X fastest, then Y, then Z).
func (vol *TSDFVolume) idx(x, y, z int) int {
	return (z*vol.dims[1]+y)*vol.dims[0] + x
}

// voxelCenter returns the world-space centre of voxel (x, y, z).
func (vol *TSDFVolume) voxelCenter(x, y, z int) [3]float64 {
	return [3]float64{
		vol.origin[0] + (float64(x)+0.5)*vol.voxelSize,
		vol.origin[1] + (float64(y)+0.5)*vol.voxelSize,
		vol.origin[2] + (float64(z)+0.5)*vol.voxelSize,
	}
}

// Integrate fuses one depth frame into the volume. camPose is the camera's
// extrinsic transform mapping a world point into the camera frame
// (p_cam = R·p_world + T); pass [IdentityPose] for a camera at the world origin
// looking down +Z. Every voxel centre is projected into the frame, and where it
// falls on a valid measurement its truncated signed distance (observed depth
// minus voxel depth, clamped to ±truncation and normalised) is blended into the
// running average with unit weight.
//
// It panics if depth is nil/empty or K has a zero focal length.
func (vol *TSDFVolume) Integrate(depth *cv.FloatMat, k [3][3]float64, camPose Pose) {
	if depth == nil || len(depth.Data) == 0 {
		panic("rgbd: TSDFVolume.Integrate given an empty depth map")
	}
	validK(k)
	fx, fy := k[0][0], k[1][1]
	cx, cy := k[0][2], k[1][2]
	rows, cols := depth.Rows, depth.Cols
	for z := 0; z < vol.dims[2]; z++ {
		for y := 0; y < vol.dims[1]; y++ {
			for x := 0; x < vol.dims[0]; x++ {
				world := vol.voxelCenter(x, y, z)
				cam := camPose.Apply(world)
				if cam[2] <= 0 {
					continue
				}
				u := int(math.Round(fx*cam[0]/cam[2] + cx))
				v := int(math.Round(fy*cam[1]/cam[2] + cy))
				if u < 0 || u >= cols || v < 0 || v >= rows {
					continue
				}
				zmeas := depth.At(v, u)
				if zmeas <= 0 {
					continue
				}
				// Signed distance along the ray: positive in front of the surface.
				sdf := zmeas - cam[2]
				if sdf < -vol.truncation {
					continue // occluded: voxel is well behind the surface
				}
				tsdf := sdf / vol.truncation
				if tsdf > 1 {
					tsdf = 1
				}
				i := vol.idx(x, y, z)
				w := vol.weight[i]
				vol.tsdf[i] = (vol.tsdf[i]*w + tsdf) / (w + 1)
				vol.weight[i] = w + 1
			}
		}
	}
}

// at returns the stored TSDF value and weight at integer voxel coordinates, or
// (0, 0) when out of range.
func (vol *TSDFVolume) at(x, y, z int) (float64, float64) {
	if x < 0 || x >= vol.dims[0] || y < 0 || y >= vol.dims[1] || z < 0 || z >= vol.dims[2] {
		return 0, 0
	}
	i := vol.idx(x, y, z)
	return vol.tsdf[i], vol.weight[i]
}

// sampleTSDF trilinearly samples the volume at a world point, returning the
// interpolated TSDF and ok=false if the point is outside the grid or lacks
// observed (weighted) support at its corners.
func (vol *TSDFVolume) sampleTSDF(world [3]float64) (float64, bool) {
	// Continuous voxel coordinate of the point (voxel centres at half-integers).
	gx := (world[0]-vol.origin[0])/vol.voxelSize - 0.5
	gy := (world[1]-vol.origin[1])/vol.voxelSize - 0.5
	gz := (world[2]-vol.origin[2])/vol.voxelSize - 0.5
	x0 := int(math.Floor(gx))
	y0 := int(math.Floor(gy))
	z0 := int(math.Floor(gz))
	fx := gx - float64(x0)
	fy := gy - float64(y0)
	fz := gz - float64(z0)
	var acc, wsum float64
	for dz := 0; dz <= 1; dz++ {
		for dy := 0; dy <= 1; dy++ {
			for dx := 0; dx <= 1; dx++ {
				val, w := vol.at(x0+dx, y0+dy, z0+dz)
				if w <= 0 {
					return 0, false
				}
				wx := fx
				if dx == 0 {
					wx = 1 - fx
				}
				wy := fy
				if dy == 0 {
					wy = 1 - fy
				}
				wz := fz
				if dz == 0 {
					wz = 1 - fz
				}
				tw := wx * wy * wz
				acc += tw * val
				wsum += tw
			}
		}
	}
	if wsum < 1e-9 {
		return 0, false
	}
	return acc / wsum, true
}

// Raycast renders a depth map by marching a ray from the camera through each
// pixel and reporting the depth at which the interpolated TSDF crosses zero
// (the reconstructed surface). camPose is the camera extrinsic (world→camera),
// matching [TSDFVolume.Integrate]; K and the output size describe the virtual
// camera. Rays that never cross a zero (miss the surface or leave the grid) get
// depth 0.
//
// It panics if K has a zero focal length or the output size is not positive.
func (vol *TSDFVolume) Raycast(k [3][3]float64, camPose Pose, rows, cols int) *cv.FloatMat {
	validK(k)
	if rows <= 0 || cols <= 0 {
		panic("rgbd: TSDFVolume.Raycast requires a positive output size")
	}
	inv := camPose.Inverse() // camera→world
	out := cv.NewFloatMat(rows, cols)
	// March in steps of half a voxel out to the far corner of the grid.
	step := vol.voxelSize * 0.5
	diag := vol.voxelSize * math.Sqrt(float64(vol.dims[0]*vol.dims[0]+vol.dims[1]*vol.dims[1]+vol.dims[2]*vol.dims[2]))
	// Distance from the camera origin (in world) to the grid, used as a start.
	camOrigin := inv.T
	for v := 0; v < rows; v++ {
		for u := 0; u < cols; u++ {
			// Ray direction in world: back-project a unit-depth pixel and rotate.
			dirCam := backProject(u, v, 1, k)
			dirWorld := matVec3(inv.R, dirCam) // camera→world rotation only
			// March along Z (camera depth) so the reported value is metric depth.
			prevVal := 0.0
			havePrev := false
			var prevT float64
			for t := step; t <= diag+step; t += step {
				world := add3(camOrigin, scale3(dirWorld, t))
				val, ok := vol.sampleTSDF(world)
				if !ok {
					havePrev = false
					continue
				}
				if havePrev && prevVal > 0 && val <= 0 {
					// Zero crossing between prevT and t: interpolate the depth.
					frac := prevVal / (prevVal - val)
					tHit := prevT + frac*(t-prevT)
					out.Data[v*cols+u] = tHit // dirWorld has camera Z-length 1
					break
				}
				prevVal = val
				prevT = t
				havePrev = true
			}
		}
	}
	return out
}

// FetchPointCloud extracts a surface point cloud from the volume by scanning for
// sign changes of the TSDF between adjacent voxels along each axis and linearly
// interpolating the zero crossing. Only voxels with observed (positive) weight
// on both sides contribute. Points are returned in world coordinates in a
// deterministic scan order (X fastest), so repeated calls agree exactly.
func (vol *TSDFVolume) FetchPointCloud() [][3]float64 {
	var pts [][3]float64
	cross := func(x0, y0, z0, x1, y1, z1 int) {
		v0, w0 := vol.at(x0, y0, z0)
		v1, w1 := vol.at(x1, y1, z1)
		if w0 <= 0 || w1 <= 0 {
			return
		}
		if (v0 > 0 && v1 <= 0) || (v0 < 0 && v1 >= 0) {
			frac := v0 / (v0 - v1)
			c0 := vol.voxelCenter(x0, y0, z0)
			c1 := vol.voxelCenter(x1, y1, z1)
			pts = append(pts, add3(c0, scale3(sub3(c1, c0), frac)))
		}
	}
	for z := 0; z < vol.dims[2]; z++ {
		for y := 0; y < vol.dims[1]; y++ {
			for x := 0; x < vol.dims[0]; x++ {
				if x+1 < vol.dims[0] {
					cross(x, y, z, x+1, y, z)
				}
				if y+1 < vol.dims[1] {
					cross(x, y, z, x, y+1, z)
				}
				if z+1 < vol.dims[2] {
					cross(x, y, z, x, y, z+1)
				}
			}
		}
	}
	return pts
}
