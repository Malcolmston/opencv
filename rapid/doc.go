// Package rapid is a standard-library-only port of OpenCV's rapid module for
// real-time silhouette-based 3D pose tracking, built on top of the root cv
// package (github.com/malcolmston/opencv).
//
// RAPID (Real-time Attitude and Position Determination) tracks the 6-DOF pose
// of a known rigid object by aligning the projected silhouette edges of its 3D
// mesh with intensity edges in the image. Given an approximate pose it performs
// the classic one-dimensional search: control points are sampled along the
// projected contour, a short image line ("line bundle") is read out along each
// contour normal, the strongest edge along every line is located, and the
// resulting perpendicular displacements drive a Gauss-Newton update of the pose.
//
// # Dependencies
//
// The package depends only on the Go standard library and the root cv package:
// no cgo, no third-party code. All numerical kernels — Rodrigues conversions,
// the pinhole projection Jacobian, and the small symmetric linear solver used
// by the Gauss-Newton step — are implemented from scratch in this package.
//
// # Pipeline
//
// The high-level tracker mirrors the OpenCV algorithm as a sequence of reusable
// primitives:
//
//   - [ExtractControlPoints] projects the mesh at the current pose, finds the
//     silhouette (contour) edges by back-face analysis, and samples control
//     points along them, each carrying its 3D model coordinate and 2D search
//     normal.
//   - [ExtractLineBundle] reads a 1-D intensity profile of length 2*len+1 along
//     every control point's normal into a bundle image, recording the sampled
//     pixel coordinates.
//   - [FindCorrespondencies] locates the strongest gradient (edge) along each
//     bundle row and returns its column and response.
//   - [ConvertCorrespondencies] turns the found columns into matched 2D image
//     points, pairs them with their 3D model points and normals, and masks out
//     weak or out-of-bounds matches.
//
// [Rapid], [OLSTracker] and [GOSTracker] wrap these primitives behind the
// [Tracker] interface. [Rapid.Proceed] performs a single RAPID iteration and
// returns the updated pose together with the ratio of successful
// correspondences and the RMS reprojection error; [Rapid.Compute] iterates
// until a [TermCriteria] is met.
//
// The drawing helpers [DrawWireframe], [DrawSearchLines] and
// [DrawCorrespondencies] visualise the mesh, the search lines and the located
// edges respectively.
package rapid
