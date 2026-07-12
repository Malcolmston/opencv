package surface_matching

import "math"

// PoseInliers counts how many model points, after being mapped into the scene
// by pose, land within inlierDist of some scene point. It is the geometric
// support of a hypothesised pose and the basis of [ScorePose] and [VerifyPose].
// The scene search is accelerated with a [KDTree3D]. inlierDist must be
// positive; it panics otherwise or on empty clouds.
func PoseInliers(model, scene *PointCloud, pose Pose3D, inlierDist float64) int {
	if model == nil || model.Len() == 0 || scene == nil || scene.Len() == 0 {
		panic("surface_matching: PoseInliers requires non-empty clouds")
	}
	if inlierDist <= 0 {
		panic("surface_matching: PoseInliers requires inlierDist > 0")
	}
	tree := NewKDTree3D(scene.Points)
	thresh := inlierDist * inlierDist
	count := 0
	for _, p := range model.Points {
		_, d2 := tree.Nearest(pose.Apply(p))
		if d2 <= thresh {
			count++
		}
	}
	return count
}

// ScorePose returns the fraction of model points that are inliers under pose
// (see [PoseInliers]), a value in [0, 1] where 1 means every model point has a
// scene point within inlierDist. It is a cheap, normal-free verification score
// for ranking or accepting pose hypotheses from [PPF3DDetector.Match].
func ScorePose(model, scene *PointCloud, pose Pose3D, inlierDist float64) float64 {
	n := model.Len()
	if n == 0 {
		return 0
	}
	return float64(PoseInliers(model, scene, pose, inlierDist)) / float64(n)
}

// ScorePoseNormals is a stricter verification that counts a model point as an
// inlier only when its nearest scene point is both within inlierDist and has a
// compatible surface normal — the angle between the pose-rotated model normal
// and the scene normal is at most normalThresh radians. It returns the inlier
// fraction in [0, 1]. Both clouds must carry normals. It panics on empty clouds,
// non-positive inlierDist, or missing normals.
func ScorePoseNormals(model, scene *PointCloud, pose Pose3D, inlierDist, normalThresh float64) float64 {
	if model == nil || model.Len() == 0 || scene == nil || scene.Len() == 0 {
		panic("surface_matching: ScorePoseNormals requires non-empty clouds")
	}
	if inlierDist <= 0 {
		panic("surface_matching: ScorePoseNormals requires inlierDist > 0")
	}
	if len(model.Normals) != len(model.Points) || len(scene.Normals) != len(scene.Points) {
		panic("surface_matching: ScorePoseNormals requires per-point normals on both clouds")
	}
	tree := NewKDTree3D(scene.Points)
	thresh := inlierDist * inlierDist
	cosThresh := math.Cos(normalThresh)
	count := 0
	for i, p := range model.Points {
		j, d2 := tree.Nearest(pose.Apply(p))
		if d2 > thresh {
			continue
		}
		mn := normalize3(pose.ApplyNormal(model.Normals[i]))
		sn := normalize3(scene.Normals[j])
		if dot3(mn, sn) >= cosThresh {
			count++
		}
	}
	return float64(count) / float64(model.Len())
}

// VerifyPose reports whether pose is acceptable, i.e. its [ScorePose] inlier
// fraction is at least minInlierRatio, and returns that score alongside the
// boolean. Use it to reject spurious PPF candidates before or after ICP
// refinement.
func VerifyPose(model, scene *PointCloud, pose Pose3D, inlierDist, minInlierRatio float64) (bool, float64) {
	score := ScorePose(model, scene, pose, inlierDist)
	return score >= minInlierRatio, score
}
