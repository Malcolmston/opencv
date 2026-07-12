package ccalib

// This file provides a MultiCameraCalibration-style helper: recovering the
// relative poses of several rigidly-mounted omnidirectional cameras and of the
// calibration-pattern instances they jointly observe, by pose-graph
// propagation. It mirrors the spirit of OpenCV's cv::multicalib module.

// Pose is a rigid transform: a rotation R and translation T that map a point p
// to R·p + T. The zero value is the identity rotation with zero translation
// only after [NewPose]; use [IdentityPose] for a valid identity.
type Pose struct {
	R [3][3]float64
	T [3]float64
}

// IdentityPose returns the identity rigid transform.
func IdentityPose() Pose {
	return Pose{R: [3][3]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}}
}

// Apply returns R·p + T.
func (p Pose) Apply(v [3]float64) [3]float64 { return add3(matVec3(p.R, v), p.T) }

// Compose returns the transform equivalent to applying q first, then p
// (p ∘ q): (p·q).R = p.R·q.R, (p·q).T = p.R·q.T + p.T.
func (p Pose) Compose(q Pose) Pose {
	return Pose{R: mul3(p.R, q.R), T: add3(matVec3(p.R, q.T), p.T)}
}

// Inverse returns the inverse rigid transform.
func (p Pose) Inverse() Pose {
	rt := transpose3(p.R)
	return Pose{R: rt, T: scale3(matVec3(rt, p.T), -1)}
}

// Rvec returns the rotation as an axis-angle vector.
func (p Pose) Rvec() [3]float64 { return rodriguesToVector(p.R) }

// MultiCameraObservation records one camera's view of one pattern instance:
// the pattern's model points (ObjectPoints, with Z = 0) and their observed
// pixels (ImagePoints) in camera Camera during frame Frame.
type MultiCameraObservation struct {
	Camera       int
	Frame        int
	ObjectPoints [][3]float64
	ImagePoints  [][2]float64
}

// MultiCameraResult holds the recovered geometry. CameraPoses[c] maps a point
// in the reference frame (camera 0) into camera c's frame — CameraPoses[0] is
// the identity. FramePoses[f] maps a point in pattern f's local frame into the
// reference frame. CameraKnown[c] and FrameKnown[f] report whether that pose
// was reachable from the observation graph.
type MultiCameraResult struct {
	CameraPoses []Pose
	FramePoses  []Pose
	CameraKnown []bool
	FrameKnown  []bool
}

// MultiCameraCalibration recovers the relative poses of numCameras
// omnidirectional cameras and numFrames pattern instances from their
// observations. models[c] is camera c's (known) intrinsic model; obs lists every
// (camera, frame) observation. Camera 0 defines the reference frame.
//
// Each observation is reduced to the pattern-to-camera pose P(c,f) with
// [solvePose]. Treating cameras and frames as the two sides of a bipartite pose
// graph (edge P(c,f) = CameraPose(c) ∘ FramePose(f)), the routine fixes camera 0
// to the identity and propagates poses breadth-first across the graph, so every
// camera and frame connected to the reference is resolved. ok is false when the
// input is inconsistent or no observation involves camera 0.
func (omnidirNS) MultiCameraCalibration(models []OmniModel, numFrames int, obs []MultiCameraObservation) (MultiCameraResult, bool) {
	numCameras := len(models)
	if numCameras == 0 || numFrames == 0 {
		return MultiCameraResult{}, false
	}
	// Reduce observations to pattern-to-camera poses.
	type edge struct {
		cam, frame int
		pose       Pose
	}
	var edges []edge
	adjCam := make([][]int, numCameras)  // camera -> edge indices
	adjFrame := make([][]int, numFrames) // frame -> edge indices
	for _, o := range obs {
		if o.Camera < 0 || o.Camera >= numCameras || o.Frame < 0 || o.Frame >= numFrames {
			continue
		}
		rvec, tvec, okp := solvePose(o.ObjectPoints, o.ImagePoints, models[o.Camera])
		if !okp {
			continue
		}
		e := edge{cam: o.Camera, frame: o.Frame, pose: Pose{R: rodriguesToMatrix(rvec), T: tvec}}
		idx := len(edges)
		edges = append(edges, e)
		adjCam[o.Camera] = append(adjCam[o.Camera], idx)
		adjFrame[o.Frame] = append(adjFrame[o.Frame], idx)
	}
	if len(edges) == 0 {
		return MultiCameraResult{}, false
	}
	res := MultiCameraResult{
		CameraPoses: make([]Pose, numCameras),
		FramePoses:  make([]Pose, numFrames),
		CameraKnown: make([]bool, numCameras),
		FrameKnown:  make([]bool, numFrames),
	}
	// BFS from camera 0. Node encoding: cameras 0..numCameras-1, frames offset by
	// numCameras. P(c,f) = CameraPose(c) ∘ FramePose(f), so:
	//   FramePose(f) = CameraPose(c)^{-1} ∘ P(c,f)
	//   CameraPose(c) = P(c,f) ∘ FramePose(f)^{-1}
	res.CameraPoses[0] = IdentityPose()
	res.CameraKnown[0] = true
	queue := []int{0}
	for len(queue) > 0 {
		node := queue[0]
		queue = queue[1:]
		if node < numCameras {
			c := node
			for _, ei := range adjCam[c] {
				e := edges[ei]
				if res.FrameKnown[e.frame] {
					continue
				}
				res.FramePoses[e.frame] = res.CameraPoses[c].Inverse().Compose(e.pose)
				res.FrameKnown[e.frame] = true
				queue = append(queue, numCameras+e.frame)
			}
		} else {
			f := node - numCameras
			for _, ei := range adjFrame[f] {
				e := edges[ei]
				if res.CameraKnown[e.cam] {
					continue
				}
				res.CameraPoses[e.cam] = e.pose.Compose(res.FramePoses[f].Inverse())
				res.CameraKnown[e.cam] = true
				queue = append(queue, e.cam)
			}
		}
	}
	return res, true
}
