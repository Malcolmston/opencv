package surface_matching

import (
	"bufio"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
)

// PointCloud is an oriented point cloud: a slice of 3-D points and a parallel
// slice of unit surface normals. Points and Normals must have equal length; the
// normal at index i belongs to the point at index i.
//
// The zero value is an empty cloud. Clouds are the fundamental input to
// [PPF3DDetector] and [ICP]; both matching and refinement require normals, so
// populate Normals directly, load them from a file, or estimate them with
// [PointCloud.ComputeNormals].
type PointCloud struct {
	Points  []Vec3
	Normals []Vec3
}

// NewPointCloud builds a cloud from parallel point and normal slices. It panics
// if the slices differ in length. The slices are retained, not copied.
func NewPointCloud(points, normals []Vec3) *PointCloud {
	if len(points) != len(normals) {
		panic("surface_matching: points and normals must have equal length")
	}
	return &PointCloud{Points: points, Normals: normals}
}

// Len returns the number of points in the cloud.
func (pc *PointCloud) Len() int { return len(pc.Points) }

// Clone returns a deep copy of the cloud.
func (pc *PointCloud) Clone() *PointCloud {
	out := &PointCloud{
		Points:  make([]Vec3, len(pc.Points)),
		Normals: make([]Vec3, len(pc.Normals)),
	}
	copy(out.Points, pc.Points)
	copy(out.Normals, pc.Normals)
	return out
}

// Centroid returns the arithmetic mean of the points. It returns the origin for
// an empty cloud.
func (pc *PointCloud) Centroid() Vec3 {
	if len(pc.Points) == 0 {
		return Vec3{}
	}
	var c Vec3
	for _, p := range pc.Points {
		c = add3(c, p)
	}
	return scale3(c, 1/float64(len(pc.Points)))
}

// boundingBox returns the axis-aligned minimum and maximum corners of the
// cloud. It returns two zero vectors for an empty cloud.
func (pc *PointCloud) boundingBox() (mn, mx Vec3) {
	if len(pc.Points) == 0 {
		return Vec3{}, Vec3{}
	}
	mn = pc.Points[0]
	mx = pc.Points[0]
	for _, p := range pc.Points {
		for k := 0; k < 3; k++ {
			if p[k] < mn[k] {
				mn[k] = p[k]
			}
			if p[k] > mx[k] {
				mx[k] = p[k]
			}
		}
	}
	return mn, mx
}

// Diameter returns the length of the bounding-box diagonal, the scale used to
// derive absolute distance and sampling steps from the relative steps passed to
// the detector. It returns zero for an empty cloud.
func (pc *PointCloud) Diameter() float64 {
	mn, mx := pc.boundingBox()
	return norm3(sub3(mx, mn))
}

// Transform returns a new cloud with the rigid transform applied: each point
// maps to R·p + t and each normal to R·n (translation does not affect
// directions). The receiver is left unchanged.
func (pc *PointCloud) Transform(r Mat3, t Vec3) *PointCloud {
	out := &PointCloud{
		Points:  make([]Vec3, len(pc.Points)),
		Normals: make([]Vec3, len(pc.Normals)),
	}
	for i, p := range pc.Points {
		out.Points[i] = add3(matVec3(r, p), t)
	}
	for i, n := range pc.Normals {
		out.Normals[i] = matVec3(r, n)
	}
	return out
}

// TransformPose returns a new cloud with the pose's rigid transform applied to
// every point and normal.
func (pc *PointCloud) TransformPose(p Pose3D) *PointCloud {
	return pc.Transform(p.R, p.T)
}

// VoxelDownsample reduces the cloud by averaging every point (and normal) that
// falls into the same cubic cell of a regular grid of the given side length.
// The resulting cloud has one representative per occupied cell, with normals
// re-normalised to unit length. Output order is deterministic (cells are
// visited in sorted index order), independent of input order. It panics if
// voxelSize is not positive.
//
// Downsampling before training and matching is the standard way to bound the
// cost of the brute-force point-pair enumeration and to make the feature
// quantisation meaningful.
func (pc *PointCloud) VoxelDownsample(voxelSize float64) *PointCloud {
	if voxelSize <= 0 {
		panic("surface_matching: voxel size must be positive")
	}
	if len(pc.Points) == 0 {
		return &PointCloud{}
	}
	mn, _ := pc.boundingBox()
	type cell struct {
		sumP, sumN Vec3
		count      int
	}
	type key struct{ x, y, z int }
	cells := make(map[key]*cell)
	order := make([]key, 0)
	for i, p := range pc.Points {
		k := key{
			int(math.Floor((p[0] - mn[0]) / voxelSize)),
			int(math.Floor((p[1] - mn[1]) / voxelSize)),
			int(math.Floor((p[2] - mn[2]) / voxelSize)),
		}
		c := cells[k]
		if c == nil {
			c = &cell{}
			cells[k] = c
			order = append(order, k)
		}
		c.sumP = add3(c.sumP, p)
		if i < len(pc.Normals) {
			c.sumN = add3(c.sumN, pc.Normals[i])
		}
		c.count++
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
	out := &PointCloud{
		Points:  make([]Vec3, 0, len(order)),
		Normals: make([]Vec3, 0, len(order)),
	}
	for _, k := range order {
		c := cells[k]
		inv := 1 / float64(c.count)
		out.Points = append(out.Points, scale3(c.sumP, inv))
		out.Normals = append(out.Normals, normalize3(c.sumN))
	}
	return out
}

// ComputeNormals estimates a unit surface normal at every point from the local
// tangent plane of its k nearest neighbours (principal component analysis: the
// eigenvector of the neighbourhood covariance with the smallest eigenvalue).
// Each normal is oriented so it points toward viewpoint, giving the cloud a
// consistent outward orientation for a surface seen from that vantage point.
//
// The search is brute force (O(n²)); it is intended for the modest clouds used
// in surface matching, not for large scans. It panics if k is below 2 or the
// cloud has fewer than k points. The existing Normals slice is replaced.
func (pc *PointCloud) ComputeNormals(k int, viewpoint Vec3) {
	n := len(pc.Points)
	if k < 2 {
		panic("surface_matching: ComputeNormals requires k >= 2")
	}
	if n < k {
		panic("surface_matching: ComputeNormals requires at least k points")
	}
	normals := make([]Vec3, n)
	type nd struct {
		idx int
		d   float64
	}
	dists := make([]nd, n)
	for i := 0; i < n; i++ {
		p := pc.Points[i]
		for j := 0; j < n; j++ {
			dists[j] = nd{j, sqDist(p, pc.Points[j])}
		}
		sort.Slice(dists, func(a, b int) bool { return dists[a].d < dists[b].d })
		// Mean of the k nearest (including the point itself).
		var mean Vec3
		for t := 0; t < k; t++ {
			mean = add3(mean, pc.Points[dists[t].idx])
		}
		mean = scale3(mean, 1/float64(k))
		// Covariance of the neighbourhood.
		var cov Mat3
		for t := 0; t < k; t++ {
			d := sub3(pc.Points[dists[t].idx], mean)
			for r := 0; r < 3; r++ {
				for c := 0; c < 3; c++ {
					cov[r][c] += d[r] * d[c]
				}
			}
		}
		vals, vecs := jacobiEigenSym(cov)
		// Smallest-eigenvalue eigenvector is the plane normal.
		mi := 0
		for t := 1; t < 3; t++ {
			if vals[t] < vals[mi] {
				mi = t
			}
		}
		normal := normalize3(Vec3{vecs[0][mi], vecs[1][mi], vecs[2][mi]})
		// Orient toward the viewpoint.
		if dot3(normal, sub3(viewpoint, p)) < 0 {
			normal = scale3(normal, -1)
		}
		normals[i] = normal
	}
	pc.Normals = normals
}

// sqDist returns the squared Euclidean distance between two points.
func sqDist(a, b Vec3) float64 {
	dx := a[0] - b[0]
	dy := a[1] - b[1]
	dz := a[2] - b[2]
	return dx*dx + dy*dy + dz*dz
}

// LoadPLY reads an ASCII PLY file into a point cloud. It parses the vertex
// element's x, y, z coordinates and, when present, nx, ny, nz normals. Colour
// and any other vertex properties are skipped, and non-vertex elements (faces,
// edges) are ignored. Binary PLY is not supported.
//
// If the file carries no normals the returned cloud has an empty Normals slice;
// call [PointCloud.ComputeNormals] to estimate them before matching.
func LoadPLY(path string) (*PointCloud, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return parsePLY(f)
}

// parsePLY implements the ASCII-PLY reader used by [LoadPLY], split out so it
// can be exercised without touching the filesystem.
func parsePLY(r io.Reader) (*PointCloud, error) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 1024*1024), 64*1024*1024)

	if !sc.Scan() || strings.TrimSpace(sc.Text()) != "ply" {
		return nil, fmt.Errorf("surface_matching: not a PLY file")
	}
	var (
		vertexCount int
		props       []string
		inVertex    bool
		format      string
	)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		fields := strings.Fields(line)
		if len(fields) == 0 {
			continue
		}
		switch fields[0] {
		case "format":
			if len(fields) >= 2 {
				format = fields[1]
			}
		case "comment":
			// ignore
		case "element":
			if len(fields) >= 3 && fields[1] == "vertex" {
				inVertex = true
				vertexCount, _ = strconv.Atoi(fields[2])
			} else {
				inVertex = false
			}
		case "property":
			if inVertex && len(fields) >= 3 {
				props = append(props, fields[len(fields)-1])
			}
		case "end_header":
			goto body
		}
	}
body:
	if format != "" && format != "ascii" {
		return nil, fmt.Errorf("surface_matching: only ascii PLY is supported, got %q", format)
	}
	// Map property names to column indices.
	col := map[string]int{}
	for i, p := range props {
		col[p] = i
	}
	xi, xok := col["x"]
	yi, yok := col["y"]
	zi, zok := col["z"]
	if !xok || !yok || !zok {
		return nil, fmt.Errorf("surface_matching: PLY vertex is missing x/y/z properties")
	}
	nxi, nxok := col["nx"]
	nyi, nyok := col["ny"]
	nzi, nzok := col["nz"]
	hasNormals := nxok && nyok && nzok

	pc := &PointCloud{}
	for i := 0; i < vertexCount && sc.Scan(); i++ {
		f := strings.Fields(sc.Text())
		if len(f) < len(props) {
			return nil, fmt.Errorf("surface_matching: PLY vertex line %d has too few fields", i)
		}
		p, err := parseVec(f, xi, yi, zi)
		if err != nil {
			return nil, err
		}
		pc.Points = append(pc.Points, p)
		if hasNormals {
			nrm, err := parseVec(f, nxi, nyi, nzi)
			if err != nil {
				return nil, err
			}
			pc.Normals = append(pc.Normals, normalize3(nrm))
		}
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	return pc, nil
}

// parseVec reads three float columns from a whitespace-split PLY line.
func parseVec(f []string, i, j, k int) (Vec3, error) {
	x, err := strconv.ParseFloat(f[i], 64)
	if err != nil {
		return Vec3{}, err
	}
	y, err := strconv.ParseFloat(f[j], 64)
	if err != nil {
		return Vec3{}, err
	}
	z, err := strconv.ParseFloat(f[k], 64)
	if err != nil {
		return Vec3{}, err
	}
	return Vec3{x, y, z}, nil
}
