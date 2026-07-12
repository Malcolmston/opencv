package surface_matching

// ClusterPoses groups poses whose rotations agree within angleThresh radians and
// whose translations agree within transThresh, then replaces each group by a
// single pose that is the vote-weighted average of its members (rotations
// averaged as quaternions, translations as a weighted mean) with the group's
// summed Votes. The result is sorted by descending Votes.
//
// It is the exported form of the clustering the detector applies internally, so
// callers assembling candidate poses from several sources — multiple detectors,
// or [PPF3DDetector.Match] runs over scene tiles — can consolidate them with the
// same rotation-plus-translation proximity rule. It is deterministic for a given
// input order.
func ClusterPoses(poses []Pose3D, angleThresh, transThresh float64) []Pose3D {
	return clusterPoses(poses, angleThresh, transThresh)
}

// AveragePoses returns a single pose that averages the given poses: a
// vote-weighted quaternion mean of the rotations (aligned to a common
// hemisphere to avoid antipodal cancellation), a vote-weighted mean of the
// translations, and the summed Votes. It returns the zero pose for an empty
// slice. This is the averaging primitive underlying [ClusterPoses].
func AveragePoses(poses []Pose3D) Pose3D {
	if len(poses) == 0 {
		return Pose3D{}
	}
	return averagePoses(poses)
}

// SuppressNonMaximum performs greedy non-maximum suppression over a set of pose
// hypotheses: considering poses in descending vote order, it keeps a pose only
// if it is not within (angleThresh, transThresh) of an already-kept, stronger
// pose. Unlike [ClusterPoses] it neither averages nor merges votes — each
// survivor is an original pose — which makes it the right tool for reporting
// several distinct object instances in one scene. The kept poses are returned in
// descending vote order.
func SuppressNonMaximum(poses []Pose3D, angleThresh, transThresh float64) []Pose3D {
	if len(poses) == 0 {
		return nil
	}
	sorted := make([]Pose3D, len(poses))
	copy(sorted, poses)
	insertionSortByVotes(sorted)

	kept := make([]Pose3D, 0, len(sorted))
	for _, ps := range sorted {
		suppressed := false
		for _, k := range kept {
			if rotationAngle(k.R, ps.R) <= angleThresh &&
				norm3(sub3(k.T, ps.T)) <= transThresh {
				suppressed = true
				break
			}
		}
		if !suppressed {
			kept = append(kept, ps)
		}
	}
	return kept
}
