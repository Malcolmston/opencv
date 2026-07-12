package surface_matching

import (
	"math"
	"sort"
)

// PPFHashIndex is a compact, immutable, cache-friendly view of a trained
// detector's point-pair-feature table. It flattens the training hash map into
// three parallel arrays — sorted feature keys, per-key offsets, and one
// contiguous run of pair records — so a lookup is a binary search plus a slice,
// with none of the pointer chasing of a map of slices. It accelerates the tight
// voting inner loop of [PPF3DDetector.MatchInstances].
//
// Build one from a trained detector with [PPF3DDetector.BuildHashIndex]; it is
// read-only and safe to share across queries.
type PPFHashIndex struct {
	keys    []uint64
	offsets []int
	infos   []ppfInfo
}

// BuildHashIndex flattens the detector's trained hash table into a
// [PPFHashIndex]. It panics if the detector is untrained.
func (d *PPF3DDetector) BuildHashIndex() *PPFHashIndex {
	if !d.trained {
		panic("surface_matching: BuildHashIndex called before TrainModel")
	}
	keys := make([]uint64, 0, len(d.hashTable))
	total := 0
	for k, v := range d.hashTable {
		keys = append(keys, k)
		total += len(v)
	}
	sort.Slice(keys, func(i, j int) bool { return keys[i] < keys[j] })
	idx := &PPFHashIndex{
		keys:    keys,
		offsets: make([]int, len(keys)+1),
		infos:   make([]ppfInfo, 0, total),
	}
	for i, k := range keys {
		idx.offsets[i] = len(idx.infos)
		idx.infos = append(idx.infos, d.hashTable[k]...)
	}
	idx.offsets[len(keys)] = len(idx.infos)
	return idx
}

// Len returns the number of distinct feature keys in the index.
func (idx *PPFHashIndex) Len() int { return len(idx.keys) }

// Lookup returns the pair records stored under key, or nil if the key is absent.
// The returned slice aliases the index's storage; treat it as read-only.
func (idx *PPFHashIndex) Lookup(key uint64) []ppfInfo {
	lo, hi := 0, len(idx.keys)
	for lo < hi {
		mid := (lo + hi) / 2
		if idx.keys[mid] < key {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	if lo >= len(idx.keys) || idx.keys[lo] != key {
		return nil
	}
	return idx.infos[idx.offsets[lo]:idx.offsets[lo+1]]
}

// quantizePPF returns the integer quantised components of a point-pair feature,
// matching the packing used by [PPF3DDetector.hashKey].
func (d *PPF3DDetector) quantizePPF(f1, f2, f3, f4 float64) (int64, int64, int64, int64) {
	featAngleStep := math.Pi / float64(d.numAngles)
	return int64(f1 / d.distStep),
		int64(f2 / featAngleStep),
		int64(f3 / featAngleStep),
		int64(f4 / featAngleStep)
}

// packPPFKey packs four quantised feature components into the 64-bit key layout
// (distance in the low lane, then the three angles), identical to
// [PPF3DDetector.hashKey].
func packPPFKey(q1, q2, q3, q4 int64) uint64 {
	return (uint64(q1) & 0xFFFF) | (uint64(q2)&0xFFFF)<<16 |
		(uint64(q3)&0xFFFF)<<32 | (uint64(q4)&0xFFFF)<<48
}

// TrainModelSpread trains the detector like [PPF3DDetector.TrainModel] but
// additionally inserts every model pair into the immediately neighbouring
// quantisation bins (±1 along each of the four feature dimensions
// independently). This "spread" indexing makes voting robust to features that
// straddle a bin boundary — a common cause of missed votes under noise — at the
// cost of a larger table. After training, ordinary [PPF3DDetector.Match] and
// [PPF3DDetector.MatchInstances] transparently benefit from the denser index.
//
// It has the same requirements and panics as [PPF3DDetector.TrainModel] and
// retrains from scratch.
func (d *PPF3DDetector) TrainModelSpread(model *PointCloud) {
	if model == nil || model.Len() < 2 {
		panic("surface_matching: TrainModelSpread requires a model with >= 2 points")
	}
	if len(model.Normals) != len(model.Points) {
		panic("surface_matching: TrainModelSpread requires per-point normals")
	}
	diameter := model.Diameter()
	if diameter <= 0 {
		panic("surface_matching: degenerate model (zero diameter)")
	}
	sampleStep := d.samplingStepRelative * diameter
	sampled := model.VoxelDownsample(sampleStep)
	if sampled.Len() < 2 {
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
			alpha := computeAlpha(refR, refT, p2)
			info := ppfInfo{refIndex: i, alpha: alpha}
			for _, key := range d.spreadKeys(f1, f2, f3, f4) {
				d.hashTable[key] = append(d.hashTable[key], info)
			}
		}
	}
	d.trained = true
}

// spreadKeys returns the centre feature key together with the neighbouring keys
// obtained by shifting each quantised dimension by ±1 (skipping shifts that go
// negative). Distinct keys are returned without duplicates.
func (d *PPF3DDetector) spreadKeys(f1, f2, f3, f4 float64) []uint64 {
	q1, q2, q3, q4 := d.quantizePPF(f1, f2, f3, f4)
	base := [4]int64{q1, q2, q3, q4}
	seen := make(map[uint64]struct{}, 9)
	keys := make([]uint64, 0, 9)
	add := func(q [4]int64) {
		k := packPPFKey(q[0], q[1], q[2], q[3])
		if _, ok := seen[k]; ok {
			return
		}
		seen[k] = struct{}{}
		keys = append(keys, k)
	}
	add(base)
	for dim := 0; dim < 4; dim++ {
		for _, delta := range [2]int64{-1, 1} {
			q := base
			q[dim] += delta
			if q[dim] < 0 {
				continue
			}
			add(q)
		}
	}
	return keys
}

// MatchInstances detects up to maxInstances distinct occurrences of the trained
// model in the scene and returns their poses in descending vote order. It votes
// exactly as [PPF3DDetector.Match] does — sampling scene reference points and
// accumulating Hough votes through the (optionally spread) hash index — but
// instead of collapsing everything into one clustered list it keeps every
// reference point's peak, clusters proximate peaks with vote-weighted averaging,
// and then applies non-maximum suppression so that only spatially separated
// instances survive. This recovers several copies of the same object placed in
// one scene, which a single-peak match cannot.
//
// relSampleStep and relDistanceStep behave as in [PPF3DDetector.Match] and must
// lie in (0, 1]; maxInstances must be positive. Two instances count as distinct
// when their translations differ by more than half the model diameter. It
// panics if the detector is untrained.
func (d *PPF3DDetector) MatchInstances(scene *PointCloud, relSampleStep, relDistanceStep float64, maxInstances int) []Pose3D {
	if !d.trained {
		panic("surface_matching: MatchInstances called before TrainModel")
	}
	if maxInstances < 1 {
		panic("surface_matching: maxInstances must be >= 1")
	}
	candidates := d.voteCandidates(scene, relSampleStep, relDistanceStep)
	if len(candidates) == 0 {
		return nil
	}
	angleThresh := 2 * d.angleStep
	clusterTrans := 0.1 * d.modelDiameter
	clustered := clusterPoses(candidates, angleThresh, clusterTrans)

	// Separate instances: suppress clustered peaks that sit closer than half a
	// diameter to a stronger one, so each surviving pose is a different object.
	instanceTrans := 0.5 * d.modelDiameter
	instances := SuppressNonMaximum(clustered, math.Pi, instanceTrans)
	if len(instances) > maxInstances {
		instances = instances[:maxInstances]
	}
	return instances
}

// voteCandidates runs the PPF Hough voting over the scene and returns the raw,
// unclustered peak pose from every sampled reference point. It shares its
// mechanics with [PPF3DDetector.Match] but exposes the full candidate set that
// multi-instance detection needs, and drives the accumulation through a
// [PPFHashIndex] for speed.
func (d *PPF3DDetector) voteCandidates(scene *PointCloud, relSampleStep, relDistanceStep float64) []Pose3D {
	if scene == nil || scene.Len() < 2 {
		return nil
	}
	if len(scene.Normals) != len(scene.Points) {
		panic("surface_matching: MatchInstances requires per-point scene normals")
	}
	if relSampleStep <= 0 || relSampleStep > 1 {
		panic("surface_matching: relSampleStep must be in (0,1]")
	}
	if relDistanceStep <= 0 || relDistanceStep > 1 {
		panic("surface_matching: relDistanceStep must be in (0,1]")
	}

	index := d.BuildHashIndex()

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
	acc := make([][]int, nModel)
	for i := range acc {
		acc[i] = make([]int, d.numAngles)
	}
	var candidates []Pose3D

	for sr := 0; sr < ns; sr += refStride {
		sp := sceneSampled.Points[sr]
		sn := sceneSampled.Normals[sr]
		refR, refT := referenceFrame(sp, sn)
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
			bucket := index.Lookup(d.hashKey(f1, f2, f3, f4))
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
		bestVotes, bestModel, bestBin := 0, -1, 0
		for i := 0; i < nModel; i++ {
			row := acc[i]
			for b := 0; b < d.numAngles; b++ {
				if row[b] > bestVotes {
					bestVotes, bestModel, bestBin = row[b], i, b
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
	return candidates
}
