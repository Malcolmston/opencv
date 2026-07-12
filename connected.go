package cv

// Connectivity selects the neighbourhood used by connected-component labelling.
type Connectivity int

const (
	// Connectivity4 considers the 4 edge-adjacent neighbours (N, S, E, W).
	Connectivity4 Connectivity = 4
	// Connectivity8 also considers the 4 diagonal neighbours.
	Connectivity8 Connectivity = 8
)

// unionFind is a disjoint-set structure over provisional component labels.
type unionFind struct {
	parent []int
}

func newUnionFind() *unionFind {
	// Index 0 is reserved for the background.
	return &unionFind{parent: []int{0}}
}

func (u *unionFind) make() int {
	id := len(u.parent)
	u.parent = append(u.parent, id)
	return id
}

func (u *unionFind) find(x int) int {
	for u.parent[x] != x {
		u.parent[x] = u.parent[u.parent[x]]
		x = u.parent[x]
	}
	return x
}

func (u *unionFind) union(a, b int) {
	ra, rb := u.find(a), u.find(b)
	if ra == rb {
		return
	}
	if ra < rb {
		u.parent[rb] = ra
	} else {
		u.parent[ra] = rb
	}
}

// ConnectedComponents labels the connected foreground regions of a binary
// single-channel image with a two-pass union-find algorithm. Any non-zero
// sample is foreground. It returns a label slice in row-major order (0 marks the
// background, 1..count-1 the components) together with the total number of
// labels including the background. conn selects 4- or 8-connectivity. It panics
// if src is not single-channel or conn is invalid.
func ConnectedComponents(src *Mat, conn Connectivity) (labels []int, count int) {
	requireChannels(src, 1, "ConnectedComponents")
	if conn != Connectivity4 && conn != Connectivity8 {
		panic("cv: ConnectedComponents connectivity must be 4 or 8")
	}
	rows, cols := src.Rows, src.Cols
	prov := make([]int, rows*cols)
	uf := newUnionFind()

	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			idx := y*cols + x
			if src.Data[idx] == 0 {
				continue
			}
			// Gather already-labelled neighbours behind the scan.
			var neigh []int
			if x > 0 && prov[idx-1] != 0 {
				neigh = append(neigh, prov[idx-1])
			}
			if y > 0 && prov[idx-cols] != 0 {
				neigh = append(neigh, prov[idx-cols])
			}
			if conn == Connectivity8 {
				if y > 0 && x > 0 && prov[idx-cols-1] != 0 {
					neigh = append(neigh, prov[idx-cols-1])
				}
				if y > 0 && x < cols-1 && prov[idx-cols+1] != 0 {
					neigh = append(neigh, prov[idx-cols+1])
				}
			}
			if len(neigh) == 0 {
				prov[idx] = uf.make()
				continue
			}
			min := neigh[0]
			for _, l := range neigh[1:] {
				if l < min {
					min = l
				}
			}
			prov[idx] = min
			for _, l := range neigh {
				uf.union(min, l)
			}
		}
	}

	// Second pass: flatten to consecutive labels.
	remap := make(map[int]int)
	next := 1
	labels = make([]int, rows*cols)
	for i, p := range prov {
		if p == 0 {
			continue
		}
		root := uf.find(p)
		lbl, ok := remap[root]
		if !ok {
			lbl = next
			remap[root] = lbl
			next++
		}
		labels[i] = lbl
	}
	return labels, next
}

// ComponentStats summarises one connected component: its bounding box, pixel
// area and centroid, mirroring the per-label rows of OpenCV's
// connectedComponentsWithStats.
type ComponentStats struct {
	// Label is the component's index (0 is the background).
	Label int
	// Rect is the axis-aligned bounding box of the component.
	Rect Rect
	// Area is the number of pixels in the component.
	Area int
	// CentroidX and CentroidY are the mean pixel coordinates of the component.
	CentroidX float64
	CentroidY float64
}

// ConnectedComponentsWithStats labels a binary single-channel image like
// [ConnectedComponents] and additionally returns per-label statistics (bounding
// box, area and centroid). The returned stats slice has one entry per label,
// index 0 describing the background. conn selects 4- or 8-connectivity.
func ConnectedComponentsWithStats(src *Mat, conn Connectivity) (labels []int, count int, stats []ComponentStats) {
	labels, count = ConnectedComponents(src, conn)
	cols := src.Cols
	stats = make([]ComponentStats, count)
	sumX := make([]float64, count)
	sumY := make([]float64, count)
	for l := range stats {
		stats[l].Label = l
		stats[l].Rect = Rect{X: src.Cols, Y: src.Rows} // sentinel for min tracking
	}
	for i, l := range labels {
		x := i % cols
		y := i / cols
		s := &stats[l]
		if s.Area == 0 {
			s.Rect.X, s.Rect.Y = x, y
			s.Rect.Width, s.Rect.Height = x, y // temporarily store max
		} else {
			if x < s.Rect.X {
				s.Rect.X = x
			}
			if y < s.Rect.Y {
				s.Rect.Y = y
			}
			if x > s.Rect.Width {
				s.Rect.Width = x
			}
			if y > s.Rect.Height {
				s.Rect.Height = y
			}
		}
		s.Area++
		sumX[l] += float64(x)
		sumY[l] += float64(y)
	}
	for l := range stats {
		s := &stats[l]
		if s.Area == 0 {
			s.Rect = Rect{}
			continue
		}
		maxX, maxY := s.Rect.Width, s.Rect.Height
		s.Rect.Width = maxX - s.Rect.X + 1
		s.Rect.Height = maxY - s.Rect.Y + 1
		s.CentroidX = sumX[l] / float64(s.Area)
		s.CentroidY = sumY[l] / float64(s.Area)
	}
	return labels, count, stats
}
