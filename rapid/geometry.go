package rapid

// project maps a single world point through pose and intrinsics K into the
// image, returning the pixel coordinate, the camera-space point, and whether
// the point lies in front of the camera.
func project(obj [3]float64, r [3][3]float64, t [3]float64, fxfycxcy [4]float64) (Point2f, [3]float64, bool) {
	xc := [3]float64{
		r[0][0]*obj[0] + r[0][1]*obj[1] + r[0][2]*obj[2] + t[0],
		r[1][0]*obj[0] + r[1][1]*obj[1] + r[1][2]*obj[2] + t[1],
		r[2][0]*obj[0] + r[2][1]*obj[1] + r[2][2]*obj[2] + t[2],
	}
	if xc[2] <= 1e-6 {
		return Point2f{}, xc, false
	}
	fx, fy, cx, cy := fxfycxcy[0], fxfycxcy[1], fxfycxcy[2], fxfycxcy[3]
	return Point2f{X: fx*xc[0]/xc[2] + cx, Y: fy*xc[1]/xc[2] + cy}, xc, true
}

// intrinsics unpacks K into the (fx, fy, cx, cy) tuple used internally.
func intrinsics(k [3][3]float64) [4]float64 {
	return [4]float64{k[0][0], k[1][1], k[0][2], k[1][2]}
}

// projectionJacobian returns the 2×6 Jacobian of the image projection of a
// world point with respect to the pose increment (dω, dt), where dω is a
// left-multiplied incremental rotation and dt an additive translation. xc is
// the camera-space point and rx = R·X = xc - t is the rotated (untranslated)
// point.
//
// The chain rule combines d(proj)/d(xc) (2×3) with d(xc)/d(params) (3×6):
//
//	d(xc)/d(dω) = -skew(rx)
//	d(xc)/d(dt) =  I
func projectionJacobian(xc, rx [3]float64, fxfycxcy [4]float64) [2][6]float64 {
	fx, fy := fxfycxcy[0], fxfycxcy[1]
	z := xc[2]
	invZ := 1 / z
	invZ2 := invZ * invZ
	// d(proj)/d(xc), 2×3.
	du := [3]float64{fx * invZ, 0, -fx * xc[0] * invZ2}
	dv := [3]float64{0, fy * invZ, -fy * xc[1] * invZ2}
	// d(xc)/d(dω) = -skew(rx):
	//   [   0   rz  -ry ]
	//   [ -rz    0   rx ]
	//   [  ry  -rx    0 ]
	ns := [3][3]float64{
		{0, rx[2], -rx[1]},
		{-rx[2], 0, rx[0]},
		{rx[1], -rx[0], 0},
	}
	var j [2][6]float64
	for c := 0; c < 3; c++ {
		// Rotation columns: du · ns[:,c].
		j[0][c] = du[0]*ns[0][c] + du[1]*ns[1][c] + du[2]*ns[2][c]
		j[1][c] = dv[0]*ns[0][c] + dv[1]*ns[1][c] + dv[2]*ns[2][c]
		// Translation columns: du · I[:,c] = du[c].
		j[0][3+c] = du[c]
		j[1][3+c] = dv[c]
	}
	return j
}

// ProjectVertices projects every vertex of the mesh through the given pose and
// intrinsics, returning image coordinates in vertex order. Vertices behind the
// camera are still returned (via their extrapolated coordinate) so callers can
// index by vertex; use [DrawWireframe] to render edges between them.
func ProjectVertices(mesh *Mesh, pose Pose, k [3][3]float64) []Point2f {
	r := rodrigues(pose.Rvec)
	kk := intrinsics(k)
	out := make([]Point2f, len(mesh.Vertices))
	for i, v := range mesh.Vertices {
		p, _, _ := project(v, r, pose.Tvec, kk)
		out[i] = p
	}
	return out
}
