package surface_matching

import (
	"fmt"
	"math"
)

// Pose3D is a rigid transform in 3-D: a rotation matrix R and a translation
// vector T that together map a model point p to R·p + T. It is the output of
// [PPF3DDetector.Match] and the input and output of [ICP].
//
// Votes carries the accumulator support that produced the pose during Hough
// voting (higher is stronger evidence); after clustering it is the summed
// support of the merged poses. Residual is the root-mean-square alignment error
// when the pose comes from [ICP], and zero otherwise.
type Pose3D struct {
	R        Mat3
	T        Vec3
	Votes    int
	Residual float64
}

// IdentityPose returns the identity transform (no rotation, no translation).
func IdentityPose() Pose3D {
	return Pose3D{R: identity3()}
}

// NewPose builds a pose from a rotation matrix and translation vector.
func NewPose(r Mat3, t Vec3) Pose3D {
	return Pose3D{R: r, T: t}
}

// Apply maps a point through the pose, returning R·p + T.
func (p Pose3D) Apply(pt Vec3) Vec3 {
	return add3(matVec3(p.R, pt), p.T)
}

// ApplyNormal maps a direction (such as a surface normal) through the pose's
// rotation only; translation does not affect directions.
func (p Pose3D) ApplyNormal(n Vec3) Vec3 {
	return matVec3(p.R, n)
}

// Matrix returns the pose as a 4×4 homogeneous transform in row-major order:
//
//	[ R00 R01 R02 Tx ]
//	[ R10 R11 R12 Ty ]
//	[ R20 R21 R22 Tz ]
//	[   0   0   0  1 ]
func (p Pose3D) Matrix() [4][4]float64 {
	return [4][4]float64{
		{p.R[0][0], p.R[0][1], p.R[0][2], p.T[0]},
		{p.R[1][0], p.R[1][1], p.R[1][2], p.T[1]},
		{p.R[2][0], p.R[2][1], p.R[2][2], p.T[2]},
		{0, 0, 0, 1},
	}
}

// Inverse returns the inverse rigid transform, mapping R·p + T back to p. For a
// rotation R the inverse rotation is Rᵀ and the inverse translation is -Rᵀ·T.
func (p Pose3D) Inverse() Pose3D {
	rt := transpose3(p.R)
	return Pose3D{
		R: rt,
		T: scale3(matVec3(rt, p.T), -1),
	}
}

// Compose returns the pose equivalent to applying q first and then p, i.e. the
// transform x ↦ p(q(x)). Rotation is p.R·q.R and translation is p.R·q.T + p.T.
func (p Pose3D) Compose(q Pose3D) Pose3D {
	return Pose3D{
		R: mul3(p.R, q.R),
		T: add3(matVec3(p.R, q.T), p.T),
	}
}

// AngleTo returns the geodesic rotation angle in radians between two poses'
// rotations, a scale-free measure of their rotational disagreement.
func (p Pose3D) AngleTo(o Pose3D) float64 {
	return rotationAngle(p.R, o.R)
}

// TranslationTo returns the Euclidean distance between two poses' translations.
func (p Pose3D) TranslationTo(o Pose3D) float64 {
	return norm3(sub3(p.T, o.T))
}

// String renders the pose compactly for debugging.
func (p Pose3D) String() string {
	return fmt.Sprintf("Pose3D{votes=%d residual=%.4g T=(%.4g,%.4g,%.4g)}",
		p.Votes, p.Residual, p.T[0], p.T[1], p.T[2])
}

// clusterPoses greedily groups poses whose rotations agree within angleThresh
// radians and whose translations agree within transThresh, then replaces each
// group by a single averaged pose whose Votes is the group's total support.
// Poses are considered in descending vote order so strong candidates seed the
// clusters. The returned slice is sorted by descending Votes and is
// deterministic for a given input order.
func clusterPoses(poses []Pose3D, angleThresh, transThresh float64) []Pose3D {
	if len(poses) == 0 {
		return nil
	}
	// Stable sort by votes descending.
	sorted := make([]Pose3D, len(poses))
	copy(sorted, poses)
	insertionSortByVotes(sorted)

	type cluster struct {
		members []Pose3D
		center  Pose3D // representative used for the proximity test
	}
	var clusters []*cluster
	for _, ps := range sorted {
		placed := false
		for _, cl := range clusters {
			if rotationAngle(cl.center.R, ps.R) <= angleThresh &&
				norm3(sub3(cl.center.T, ps.T)) <= transThresh {
				cl.members = append(cl.members, ps)
				placed = true
				break
			}
		}
		if !placed {
			clusters = append(clusters, &cluster{members: []Pose3D{ps}, center: ps})
		}
	}

	out := make([]Pose3D, 0, len(clusters))
	for _, cl := range clusters {
		out = append(out, averagePoses(cl.members))
	}
	insertionSortByVotes(out)
	return out
}

// insertionSortByVotes sorts poses in place by descending Votes, breaking ties
// deterministically by translation then not reordering equal elements further.
// A stable insertion sort keeps the ordering reproducible.
func insertionSortByVotes(ps []Pose3D) {
	for i := 1; i < len(ps); i++ {
		j := i
		for j > 0 && ps[j-1].Votes < ps[j].Votes {
			ps[j-1], ps[j] = ps[j], ps[j-1]
			j--
		}
	}
}

// averagePoses returns a single pose that averages a cluster of poses. Rotations
// are averaged as vote-weighted quaternions (aligned to a common hemisphere to
// avoid antipodal cancellation) and translations as a vote-weighted mean. The
// result's Votes is the total support of the cluster.
func averagePoses(members []Pose3D) Pose3D {
	if len(members) == 1 {
		return members[0]
	}
	ref := matToQuat(members[0].R)
	var aw, ax, ay, az float64
	var tsum Vec3
	totalVotes := 0
	var wsum float64
	for _, m := range members {
		w := float64(m.Votes)
		if w <= 0 {
			w = 1
		}
		q := matToQuat(m.R)
		if q.dot(ref) < 0 {
			q = quat{-q.w, -q.x, -q.y, -q.z}
		}
		aw += w * q.w
		ax += w * q.x
		ay += w * q.y
		az += w * q.z
		tsum = add3(tsum, scale3(m.T, w))
		wsum += w
		totalVotes += m.Votes
	}
	avgQ := quat{aw, ax, ay, az}.normalized()
	return Pose3D{
		R:     quatToMat(avgQ),
		T:     scale3(tsum, 1/wsum),
		Votes: totalVotes,
	}
}

// wrapAngle maps an angle into the half-open interval [-π, π).
func wrapAngle(a float64) float64 {
	return math.Atan2(math.Sin(a), math.Cos(a))
}
