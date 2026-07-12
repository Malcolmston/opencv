package rgbd

import (
	"math"
	"sort"
)

// voxelKey identifies a voxel cell by its integer grid coordinates.
type voxelKey struct {
	x, y, z int
}

// VoxelDownsample reduces a point cloud by overlaying a regular grid of cubic
// cells with side length leaf and replacing all points that fall in a cell with
// their centroid. Cells are emitted in a deterministic order (sorted by grid
// coordinate), so the result is stable across runs. Because it averages rather
// than snaps, the overall extent of the cloud is preserved to within about one
// leaf size. It panics if leaf is not positive.
func VoxelDownsample(points [][3]float64, leaf float64) [][3]float64 {
	if leaf <= 0 {
		panic("rgbd: VoxelDownsample requires a positive leaf size")
	}
	if len(points) == 0 {
		return nil
	}
	type accum struct {
		sum [3]float64
		n   int
	}
	cells := make(map[voxelKey]*accum)
	order := make([]voxelKey, 0)
	for _, p := range points {
		key := voxelKey{
			x: int(math.Floor(p[0] / leaf)),
			y: int(math.Floor(p[1] / leaf)),
			z: int(math.Floor(p[2] / leaf)),
		}
		a := cells[key]
		if a == nil {
			a = &accum{}
			cells[key] = a
			order = append(order, key)
		}
		a.sum = add3(a.sum, p)
		a.n++
	}
	sort.Slice(order, func(i, j int) bool {
		if order[i].x != order[j].x {
			return order[i].x < order[j].x
		}
		if order[i].y != order[j].y {
			return order[i].y < order[j].y
		}
		return order[i].z < order[j].z
	})
	out := make([][3]float64, len(order))
	for i, key := range order {
		a := cells[key]
		out[i] = scale3(a.sum, 1.0/float64(a.n))
	}
	return out
}
