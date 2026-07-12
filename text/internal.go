package text

import cv "github.com/malcolmston/opencv"

// toGray returns a single-channel copy of img. A 1-channel Mat is cloned; a
// 3-channel Mat is converted with the BT.601 luma weights via cv.CvtColor. It
// panics for any other channel count.
func toGray(img *cv.Mat) *cv.Mat {
	switch img.Channels {
	case 1:
		return img.Clone()
	case 3:
		return cv.CvtColor(img, cv.ColorRGB2Gray)
	default:
		panic("text: expected a 1- or 3-channel image")
	}
}

// intUnionFind is a disjoint-set structure whose representative is always the
// smallest member index. Fixing the root to the minimum index makes every merge
// deterministic and independent of processing order.
type intUnionFind struct {
	parent []int
}

func newIntUnionFind(n int) *intUnionFind {
	p := make([]int, n)
	for i := range p {
		p[i] = i
	}
	return &intUnionFind{parent: p}
}

// find returns the representative of x, halving the path on the way up.
func (u *intUnionFind) find(x int) int {
	for u.parent[x] != x {
		u.parent[x] = u.parent[u.parent[x]]
		x = u.parent[x]
	}
	return x
}

// union merges the sets containing a and b, keeping the smaller index as root.
func (u *intUnionFind) union(a, b int) {
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

// rectArea returns the pixel area of r.
func rectArea(r cv.Rect) int {
	if r.Width <= 0 || r.Height <= 0 {
		return 0
	}
	return r.Width * r.Height
}

// rectIoU returns the intersection-over-union of two boxes, treating Width and
// Height as inclusive pixel counts. It is 0 for disjoint boxes and 1 for
// identical ones.
func rectIoU(a, b cv.Rect) float64 {
	ax2, ay2 := a.X+a.Width, a.Y+a.Height
	bx2, by2 := b.X+b.Width, b.Y+b.Height
	ix1, iy1 := maxInt(a.X, b.X), maxInt(a.Y, b.Y)
	ix2, iy2 := minInt(ax2, bx2), minInt(ay2, by2)
	iw, ih := ix2-ix1, iy2-iy1
	if iw <= 0 || ih <= 0 {
		return 0
	}
	inter := iw * ih
	uni := rectArea(a) + rectArea(b) - inter
	if uni <= 0 {
		return 0
	}
	return float64(inter) / float64(uni)
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
