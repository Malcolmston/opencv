// Package rgbd is a standard-library-only implementation of the depth-map and
// point-cloud routines from OpenCV's contrib rgbd module, built on top of the
// root cv package (github.com/malcolmston/opencv).
//
// It depends only on the Go standard library and the root cv package: no cgo,
// no third-party code, and no dependency on any sibling cv/* subpackage. Where
// the root package offers a building block — notably
// [github.com/malcolmston/opencv.FloatMat] for single-channel float grids —
// rgbd reuses it; the numerical kernels it needs beyond that (a symmetric
// 3×3 Jacobi eigensolver and a 3×3 singular value decomposition) are
// implemented locally in this package.
//
// # Scope
//
// The package covers the core geometric primitives for turning a depth image
// into a 3-D point cloud and analysing that cloud:
//
//   - [DepthTo3D] back-projects every pixel of a depth map through the pinhole
//     intrinsics into a 3-D point, producing a dense, row-major point grid.
//   - [Compute3DNormals] estimates a per-pixel surface normal from the local
//     tangent plane spanned by neighbouring back-projected points.
//   - [RegisterDepth] warps a depth map captured by one camera into the image
//     frame of a second (colour) camera given the relative pose between them.
//   - [PlaneSegmentation] extracts one or more dominant planes from a point
//     cloud with sequential RANSAC and returns a per-point label map.
//   - [ICP] aligns two point clouds with point-to-point iterative closest
//     point, recovering the rigid transform and residual error.
//   - [VoxelDownsample] reduces a point cloud by averaging the points that fall
//     into each cell of a regular voxel grid.
//
// Building on those primitives, the package also provides the registration,
// odometry and volumetric-fusion routines that make up the rest of OpenCV's
// rgbd module:
//
//   - [Pose] is the rigid-transform type shared by the routines below, with
//     [Rodrigues] / [InverseRodrigues] converting to and from the axis-angle
//     (rotation-vector) representation.
//   - [ICPPointToPlane] aligns two clouds with the point-to-plane metric, and
//     [ColoredICP] additionally uses per-point colour to break geometric
//     ambiguity.
//   - [RgbdNormals] estimates per-pixel normals by the FALS, LINE-MOD or SRI
//     method ([NewRgbdNormals]).
//   - [DepthCleaner] fills holes and despeckles a depth map, while
//     [BilateralDepthFilter] applies edge-preserving smoothing and
//     [RescaleDepth] converts depth units.
//   - [ICPOdometry], [RgbdOdometry] and [RgbdICPOdometry] recover frame-to-frame
//     camera motion from depth (geometric), intensity (photometric) or both,
//     configured by [OdometryOptions] and reported as an [OdometryResult].
//   - [WarpFrame] re-renders an intensity image and depth from a new pose, and
//     [DepthTo3dSparse] back-projects a chosen set of pixels.
//   - [RegisterDepthDistorted] extends [RegisterDepth] with a Brown–Conrady lens
//     distortion model for the colour camera.
//   - [TSDFVolume] is a lightweight KinectFusion-style truncated signed distance
//     volume with [TSDFVolume.Integrate], [TSDFVolume.Raycast] and
//     [TSDFVolume.FetchPointCloud].
//
// # Conventions
//
// Depth maps are single-channel [github.com/malcolmston/opencv.FloatMat]
// values in which entry (row v, column u) holds the metric depth Z along the
// camera's optical axis for that pixel. A depth of zero or less marks an
// invalid (missing) measurement.
//
// Intrinsic matrices are the 3×3 row-major pinhole matrix
//
//	K = [ fx  0  cx ]
//	    [  0 fy  cy ]
//	    [  0  0   1 ]
//
// with focal lengths (fx, fy) in pixels and principal point (cx, cy). Image
// coordinates use u = column (x) and v = row (y). A pixel at (u, v) with depth
// Z back-projects to the camera-frame point
//
//	X = (u - cx) * Z / fx
//	Y = (v - cy) * Z / fy
//	Z = Z
//
// so +X points right, +Y points down and +Z points away from the camera along
// the optical axis. Points are plain [3]float64 values {X, Y, Z}. Rigid
// transforms are a 3×3 rotation matrix R and a 3-vector translation t applied
// as p' = R·p + t.
//
// # Determinism
//
// All routines are deterministic. The RANSAC sampling in [PlaneSegmentation]
// draws from a fixed-seed generator so repeated calls on the same input return
// identical planes and labels; every other routine is a fixed numerical
// computation with no randomness.
//
// # Deferred
//
// The following rgbd features from OpenCV are intentionally not implemented
// here:
//
//   - The full KinectFusion pipeline with camera tracking and a hash- or
//     octree-backed large-scale volume; [TSDFVolume] covers integration,
//     raycasting and point extraction on a dense grid but not pose tracking.
//   - Colour and semantic fusion into the volume (only a scalar TSDF is stored).
//   - The Large-Scale-Direct-monocular and dynamic-fusion variants.
package rgbd
