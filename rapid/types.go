package rapid

import "math"

// Point2f is a sub-pixel image coordinate (X is the column, Y is the row). It
// complements the integer github.com/malcolmston/opencv.Point used for
// rasterised drawing.
type Point2f struct {
	X float64
	Y float64
}

// sub returns p-q.
func (p Point2f) sub(q Point2f) Point2f { return Point2f{p.X - q.X, p.Y - q.Y} }

// dot returns the dot product p·q.
func (p Point2f) dot(q Point2f) float64 { return p.X*q.X + p.Y*q.Y }

// norm returns the Euclidean length of p.
func (p Point2f) norm() float64 { return math.Hypot(p.X, p.Y) }

// Mesh is a triangulated 3D model expressed in object (world) coordinates.
//
// Vertices holds the vertex positions and Tris holds triangles as triples of
// indices into Vertices. Triangles should share a consistent winding order so
// that silhouette extraction can classify front- and back-facing faces; the
// silhouette itself is invariant to a global winding flip.
type Mesh struct {
	// Vertices are the 3D model points.
	Vertices [][3]float64
	// Tris are triangles, each a triple of indices into Vertices.
	Tris [][3]int
}

// Pose is a rigid transform from object to camera coordinates, stored as an
// axis-angle rotation vector (Rvec, a Rodrigues vector) and a translation
// (Tvec). A world point X maps to the camera point R(Rvec)·X + Tvec.
type Pose struct {
	// Rvec is the axis-angle rotation vector (Rodrigues form).
	Rvec [3]float64
	// Tvec is the translation.
	Tvec [3]float64
}

// NewCamera builds a 3×3 pinhole intrinsic matrix K from focal lengths
// (fx, fy) and principal point (cx, cy):
//
//	[ fx  0  cx ]
//	[  0 fy  cy ]
//	[  0  0   1 ]
func NewCamera(fx, fy, cx, cy float64) [3][3]float64 {
	return [3][3]float64{
		{fx, 0, cx},
		{0, fy, cy},
		{0, 0, 1},
	}
}

// TermCriteria bounds the iteration of [Rapid.Compute]. Iteration stops after
// MaxCount steps or once the pose update falls below Epsilon (measured as the
// combined norm of the rotation and translation increment), whichever comes
// first.
type TermCriteria struct {
	// MaxCount is the maximum number of RAPID iterations.
	MaxCount int
	// Epsilon is the convergence threshold on the pose increment norm.
	Epsilon float64
}

// ControlPoint is a point sampled on the projected silhouette of the mesh. It
// carries the 2D image location it projects to, the originating 3D model point,
// and the unit search normal (perpendicular to the contour) along which the
// image is searched for the true edge.
type ControlPoint struct {
	// Image is the projected 2D location of the control point.
	Image Point2f
	// Object is the corresponding 3D model point in object coordinates.
	Object [3]float64
	// Normal is the unit search direction (contour normal) in the image.
	Normal Point2f
}
