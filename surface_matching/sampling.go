package surface_matching

import (
	"math"
	"sort"
)

// TransformPCPose returns a new cloud with the pose's rigid transform applied to
// every point (p ↦ R·p + T) and every normal (n ↦ R·n). It is the free-function
// spelling of [PointCloud.TransformPose], mirroring OpenCV's transformPCPose,
// and leaves the input cloud unchanged. A nil cloud yields nil.
func TransformPCPose(pc *PointCloud, pose Pose3D) *PointCloud {
	if pc == nil {
		return nil
	}
	return pc.Transform(pose.R, pose.T)
}

// SamplePCUniform keeps every step-th point of the cloud (indices 0, step,
// 2·step, …), returning a new cloud that carries the matching normals when the
// input has them. It is the cheapest possible thinning: no averaging, no grid,
// and it preserves exact original sample positions. step must be positive; a
// step of 1 returns a full copy. It panics if step is not positive.
func SamplePCUniform(pc *PointCloud, step int) *PointCloud {
	if step <= 0 {
		panic("surface_matching: SamplePCUniform requires step >= 1")
	}
	hasN := len(pc.Normals) == len(pc.Points)
	out := &PointCloud{}
	for i := 0; i < len(pc.Points); i += step {
		out.Points = append(out.Points, pc.Points[i])
		if hasN {
			out.Normals = append(out.Normals, pc.Normals[i])
		}
	}
	return out
}

// SamplePCByQuantization subdivides the cloud's axis-aligned bounding box into a
// regular grid of (1/relStep)³ cells — each cell spans relStep of the box extent
// along its axis — and returns one representative point per occupied cell: the
// input point closest to the cell centre. Unlike [PointCloud.VoxelDownsample],
// which averages, this preserves genuine sample positions and their measured
// normals, and its cell size adapts per axis to the data's aspect ratio. This
// mirrors OpenCV's samplePCByQuantization.
//
// relStep must lie in (0, 1]. Output is emitted in sorted cell order, so the
// result is independent of input order. It panics if relStep is out of range.
func SamplePCByQuantization(pc *PointCloud, relStep float64) *PointCloud {
	if relStep <= 0 || relStep > 1 {
		panic("surface_matching: SamplePCByQuantization requires relStep in (0,1]")
	}
	if len(pc.Points) == 0 {
		return &PointCloud{}
	}
	mn, mx := pc.boundingBox()
	var cellSize Vec3
	for k := 0; k < 3; k++ {
		ext := mx[k] - mn[k]
		if ext <= 0 {
			ext = 1
		}
		cellSize[k] = ext * relStep
	}
	hasN := len(pc.Normals) == len(pc.Points)

	type key struct{ x, y, z int }
	type rep struct {
		pointIdx int
		bestD    float64
	}
	cells := make(map[key]*rep)
	order := make([]key, 0)
	for i, p := range pc.Points {
		k := key{
			int(math.Floor((p[0] - mn[0]) / cellSize[0])),
			int(math.Floor((p[1] - mn[1]) / cellSize[1])),
			int(math.Floor((p[2] - mn[2]) / cellSize[2])),
		}
		center := Vec3{
			mn[0] + (float64(k.x)+0.5)*cellSize[0],
			mn[1] + (float64(k.y)+0.5)*cellSize[1],
			mn[2] + (float64(k.z)+0.5)*cellSize[2],
		}
		d := sqDist(p, center)
		c := cells[k]
		if c == nil {
			cells[k] = &rep{pointIdx: i, bestD: d}
			order = append(order, k)
		} else if d < c.bestD {
			c.pointIdx = i
			c.bestD = d
		}
	}
	sort.Slice(order, func(a, b int) bool {
		ka, kb := order[a], order[b]
		if ka.x != kb.x {
			return ka.x < kb.x
		}
		if ka.y != kb.y {
			return ka.y < kb.y
		}
		return ka.z < kb.z
	})
	out := &PointCloud{}
	for _, k := range order {
		idx := cells[k].pointIdx
		out.Points = append(out.Points, pc.Points[idx])
		if hasN {
			out.Normals = append(out.Normals, pc.Normals[idx])
		}
	}
	return out
}
