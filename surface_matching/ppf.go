package surface_matching

import (
	"math"
	"sort"
)

// PPF3DDetector implements 3-D surface matching with Point-Pair Features (PPF)
// after Drost et al., "Model Globally, Match Locally" (CVPR 2010). Training
// hashes every oriented point pair of a model into a quantised four-dimensional
// feature together with the pair's local rotation angle (alpha); matching votes
// sampled scene pairs against that table in a Hough scheme and reads the peaks
// out as candidate rigid poses.
//
// A detector is created with [NewPPF3DDetector], trained once with
// [PPF3DDetector.TrainModel], and then queried any number of times with
// [PPF3DDetector.Match]. All steps are deterministic: sampling is grid based and
// no randomness is used.
type PPF3DDetector struct {
	samplingStepRelative float64
	distanceStepRelative float64
	numAngles            int

	trained       bool
	modelDiameter float64
	distStep      float64 // absolute F1 (distance) quantisation step
	angleStep     float64 // alpha quantisation step, 2π / numAngles
	sampledModel  *PointCloud
	hashTable     map[uint64][]ppfInfo
}

// ppfInfo records, for one hashed model pair, the index of the reference
// (first) point and the pair's model-frame alpha angle.
type ppfInfo struct {
	refIndex int
	alpha    float64
}

// DefaultNumAngles is the number of angular bins used to quantise the pair
// angles and the alpha voting dimension when a detector is created with the
// convenience constructor.
const DefaultNumAngles = 30

// NewPPF3DDetector creates a detector. relativeSamplingStep sets the model
// down-sampling grid as a fraction of the model diameter (e.g. 0.05 keeps
// roughly one point per 5%-of-diameter cell); it also fixes the scene sampling
// scale at match time. relativeDistanceStep sets the width of the distance
// feature's quantisation bin, again as a fraction of the diameter. Both must be
// in (0, 1]. The angular resolution is [DefaultNumAngles].
//
// Smaller steps give finer models and more discriminative features at higher
// cost; 0.025–0.05 is a typical range.
func NewPPF3DDetector(relativeSamplingStep, relativeDistanceStep float64) *PPF3DDetector {
	return NewPPF3DDetectorAngles(relativeSamplingStep, relativeDistanceStep, DefaultNumAngles)
}

// NewPPF3DDetectorAngles is [NewPPF3DDetector] with an explicit number of
// angular bins. numAngles must be at least 2. More bins sharpen angular
// discrimination but shrink each bin's support.
func NewPPF3DDetectorAngles(relativeSamplingStep, relativeDistanceStep float64, numAngles int) *PPF3DDetector {
	if relativeSamplingStep <= 0 || relativeSamplingStep > 1 {
		panic("surface_matching: relativeSamplingStep must be in (0,1]")
	}
	if relativeDistanceStep <= 0 || relativeDistanceStep > 1 {
		panic("surface_matching: relativeDistanceStep must be in (0,1]")
	}
	if numAngles < 2 {
		panic("surface_matching: numAngles must be >= 2")
	}
	return &PPF3DDetector{
		samplingStepRelative: relativeSamplingStep,
		distanceStepRelative: relativeDistanceStep,
		numAngles:            numAngles,
	}
}

// computePPF returns the four-dimensional point-pair feature for oriented
// points (p1, n1) and (p2, n2): F1 is the Euclidean distance |p2−p1|; F2 and F3
// are the angles between each normal and the difference vector; F4 is the angle
// between the two normals. All angles lie in [0, π].
func computePPF(p1, n1, p2, n2 Vec3) (f1, f2, f3, f4 float64) {
	d := sub3(p2, p1)
	f1 = norm3(d)
	du := normalize3(d)
	f2 = angleBetween(n1, du)
	f3 = angleBetween(n2, du)
	f4 = angleBetween(n1, n2)
	return f1, f2, f3, f4
}

// angleBetween returns the unsigned angle in [0, π] between two vectors, robust
// to floating-point drift past ±1 in the cosine.
func angleBetween(a, b Vec3) float64 {
	na := norm3(a)
	nb := norm3(b)
	if na < 1e-12 || nb < 1e-12 {
		return 0
	}
	c := dot3(a, b) / (na * nb)
	if c > 1 {
		c = 1
	}
	if c < -1 {
		c = -1
	}
	return math.Acos(c)
}

// hashKey quantises a point-pair feature into a 64-bit hash key. The distance
// occupies the low 16 bits and the three angles the next three 16-bit lanes;
// with the step sizes used here every quantised value fits comfortably in its
// lane.
func (d *PPF3DDetector) hashKey(f1, f2, f3, f4 float64) uint64 {
	featAngleStep := math.Pi / float64(d.numAngles)
	q1 := uint64(f1 / d.distStep)
	q2 := uint64(f2 / featAngleStep)
	q3 := uint64(f3 / featAngleStep)
	q4 := uint64(f4 / featAngleStep)
	return (q1 & 0xFFFF) | (q2&0xFFFF)<<16 | (q3&0xFFFF)<<32 | (q4&0xFFFF)<<48
}

// computeAlpha returns the local rotation angle of point p2 about the x axis
// after the reference frame (refR, refT) has been applied, i.e. the angle that
// rotates the transformed point into the x-y half-plane. This is the quantity
// shared between model training and scene voting.
func computeAlpha(refR Mat3, refT Vec3, p2 Vec3) float64 {
	t := add3(matVec3(refR, p2), refT)
	return math.Atan2(-t[2], t[1])
}

// referenceFrame returns the rigid transform (R, t) that sends point p to the
// origin and its normal n to the +x axis: q = R·x + t maps p to 0 and n to
// (1,0,0). It is the per-point local frame used by both training and voting.
func referenceFrame(p, n Vec3) (Mat3, Vec3) {
	r := alignToXAxis(n)
	t := scale3(matVec3(r, p), -1)
	return r, t
}

// TrainModel builds the point-pair-feature hash table from a model cloud. The
// model is first voxel-down-sampled at the detector's sampling step; then every
// ordered pair of sampled oriented points is hashed, storing the reference
// point index and the pair's model-frame alpha. The model must carry normals
// and hold at least two points after sampling. Calling TrainModel again
// retrains from scratch.
func (d *PPF3DDetector) TrainModel(model *PointCloud) {
	if model == nil || model.Len() < 2 {
		panic("surface_matching: TrainModel requires a model with >= 2 points")
	}
	if len(model.Normals) != len(model.Points) {
		panic("surface_matching: TrainModel requires per-point normals")
	}
	diameter := model.Diameter()
	if diameter <= 0 {
		panic("surface_matching: degenerate model (zero diameter)")
	}
	sampleStep := d.samplingStepRelative * diameter
	sampled := model.VoxelDownsample(sampleStep)
	if sampled.Len() < 2 {
		// Sampling collapsed the model; fall back to the raw cloud.
		sampled = model.Clone()
	}

	d.modelDiameter = diameter
	d.distStep = d.distanceStepRelative * diameter
	d.angleStep = 2 * math.Pi / float64(d.numAngles)
	d.sampledModel = sampled
	d.hashTable = make(map[uint64][]ppfInfo)

	n := sampled.Len()
	for i := 0; i < n; i++ {
		p1 := sampled.Points[i]
		n1 := sampled.Normals[i]
		refR, refT := referenceFrame(p1, n1)
		for j := 0; j < n; j++ {
			if i == j {
				continue
			}
			p2 := sampled.Points[j]
			n2 := sampled.Normals[j]
			f1, f2, f3, f4 := computePPF(p1, n1, p2, n2)
			key := d.hashKey(f1, f2, f3, f4)
			alpha := computeAlpha(refR, refT, p2)
			d.hashTable[key] = append(d.hashTable[key], ppfInfo{refIndex: i, alpha: alpha})
		}
	}
	d.trained = true
}

// Trained reports whether the detector has a hash table ready for matching.
func (d *PPF3DDetector) Trained() bool { return d.trained }

// SampledModel returns the down-sampled model cloud built during training, or
// nil if the detector is untrained. The returned cloud is the detector's own;
// treat it as read-only.
func (d *PPF3DDetector) SampledModel() *PointCloud { return d.sampledModel }

// Match aligns a scene cloud against the trained model and returns candidate
// poses that map model points into the scene, sorted by descending vote
// support. The scene is voxel-down-sampled at relDistanceStep of the model
// diameter; every (1/relSampleStep)-th sampled point is used as a Hough
// reference point and paired against all sampled scene points. For each
// reference point the accumulator over (model point, alpha) is filled, its peak
// becomes one candidate pose, and the candidates are finally clustered and
// averaged so repeated detections of the same object reinforce one another.
//
// relSampleStep and relDistanceStep must lie in (0, 1]; smaller values use more
// reference points and a finer scene, improving robustness at higher cost. The
// scene must carry normals. It panics if the detector is untrained.
func (d *PPF3DDetector) Match(scene *PointCloud, relSampleStep, relDistanceStep float64) []Pose3D {
	if !d.trained {
		panic("surface_matching: Match called before TrainModel")
	}
	if scene == nil || scene.Len() < 2 {
		return nil
	}
	if len(scene.Normals) != len(scene.Points) {
		panic("surface_matching: Match requires per-point scene normals")
	}
	if relSampleStep <= 0 || relSampleStep > 1 {
		panic("surface_matching: relSampleStep must be in (0,1]")
	}
	if relDistanceStep <= 0 || relDistanceStep > 1 {
		panic("surface_matching: relDistanceStep must be in (0,1]")
	}

	sceneSampled := scene.VoxelDownsample(relDistanceStep * d.modelDiameter)
	ns := sceneSampled.Len()
	if ns < 2 {
		sceneSampled = scene
		ns = sceneSampled.Len()
	}

	refStride := int(math.Round(1 / relSampleStep))
	if refStride < 1 {
		refStride = 1
	}

	nModel := d.sampledModel.Len()
	var candidates []Pose3D

	// Accumulator reused across reference points: rows are model points, columns
	// alpha bins.
	acc := make([][]int, nModel)
	for i := range acc {
		acc[i] = make([]int, d.numAngles)
	}

	for sr := 0; sr < ns; sr += refStride {
		sp := sceneSampled.Points[sr]
		sn := sceneSampled.Normals[sr]
		refR, refT := referenceFrame(sp, sn)

		// Clear accumulator.
		for i := 0; i < nModel; i++ {
			row := acc[i]
			for k := range row {
				row[k] = 0
			}
		}

		for si := 0; si < ns; si++ {
			if si == sr {
				continue
			}
			ip := sceneSampled.Points[si]
			in := sceneSampled.Normals[si]
			f1, f2, f3, f4 := computePPF(sp, sn, ip, in)
			key := d.hashKey(f1, f2, f3, f4)
			bucket := d.hashTable[key]
			if len(bucket) == 0 {
				continue
			}
			alphaScene := computeAlpha(refR, refT, ip)
			for _, info := range bucket {
				alpha := wrapAngle(info.alpha - alphaScene)
				bin := int((alpha + math.Pi) / d.angleStep)
				if bin < 0 {
					bin = 0
				}
				if bin >= d.numAngles {
					bin = d.numAngles - 1
				}
				acc[info.refIndex][bin]++
			}
		}

		// Peak of the accumulator.
		bestVotes := 0
		bestModel := -1
		bestBin := 0
		for i := 0; i < nModel; i++ {
			row := acc[i]
			for b := 0; b < d.numAngles; b++ {
				if row[b] > bestVotes {
					bestVotes = row[b]
					bestModel = i
					bestBin = b
				}
			}
		}
		if bestModel < 0 || bestVotes == 0 {
			continue
		}

		alpha := (float64(bestBin)+0.5)*d.angleStep - math.Pi
		pose := d.recoverPose(bestModel, refR, refT, alpha)
		pose.Votes = bestVotes
		candidates = append(candidates, pose)
	}

	if len(candidates) == 0 {
		return nil
	}

	// Cluster proximate poses so multiple reference points that agree combine
	// their support. Thresholds scale with the model.
	angleThresh := 2 * d.angleStep
	transThresh := 0.1 * d.modelDiameter
	return clusterPoses(candidates, angleThresh, transThresh)
}

// recoverPose reconstructs the model-to-scene rigid transform for a peak in the
// accumulator. With the model reference frame (R_m, t_m) for model point mIdx,
// the scene reference frame (R_s, t_s), and the voted alpha, the pose is
//
//	x_scene = R_sᵀ · ( Rx(alpha) · (R_m·x + t_m) − t_s ),
//
// giving rotation R_sᵀ·Rx(alpha)·R_m and translation R_sᵀ·(Rx(alpha)·t_m − t_s).
func (d *PPF3DDetector) recoverPose(mIdx int, sceneR Mat3, sceneT Vec3, alpha float64) Pose3D {
	mp := d.sampledModel.Points[mIdx]
	mn := d.sampledModel.Normals[mIdx]
	modelR, modelT := referenceFrame(mp, mn)

	rx := rotationX(alpha)
	rsT := transpose3(sceneR)

	rot := mul3(rsT, mul3(rx, modelR))
	trans := matVec3(rsT, sub3(matVec3(rx, modelT), sceneT))
	return Pose3D{R: rot, T: trans}
}

// hashBuckets returns the number of distinct feature keys in the trained hash
// table, a rough measure of the model's feature diversity. It returns zero for
// an untrained detector.
func (d *PPF3DDetector) hashBuckets() int {
	return len(d.hashTable)
}

// SortPosesByVotes returns a copy of poses ordered by descending vote support.
// It is exposed as a small convenience for callers assembling their own
// candidate lists.
func SortPosesByVotes(poses []Pose3D) []Pose3D {
	out := make([]Pose3D, len(poses))
	copy(out, poses)
	sort.SliceStable(out, func(i, j int) bool { return out[i].Votes > out[j].Votes })
	return out
}
