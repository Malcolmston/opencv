package stitch

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// Bounds is an axis-aligned integer rectangle on the mosaic canvas. MinX and
// MinY are inclusive; MaxX and MaxY are exclusive, so the rectangle covers
// columns [MinX, MaxX) and rows [MinY, MaxY).
type Bounds struct {
	// MinX is the inclusive left edge (smallest column).
	MinX int
	// MinY is the inclusive top edge (smallest row).
	MinY int
	// MaxX is the exclusive right edge (one past the largest column).
	MaxX int
	// MaxY is the exclusive bottom edge (one past the largest row).
	MaxY int
}

// Width returns the width of the rectangle in pixels, or 0 if empty.
func (b Bounds) Width() int {
	if b.MaxX <= b.MinX {
		return 0
	}
	return b.MaxX - b.MinX
}

// Height returns the height of the rectangle in pixels, or 0 if empty.
func (b Bounds) Height() int {
	if b.MaxY <= b.MinY {
		return 0
	}
	return b.MaxY - b.MinY
}

// Empty reports whether the rectangle encloses no pixels.
func (b Bounds) Empty() bool {
	return b.MaxX <= b.MinX || b.MaxY <= b.MinY
}

// Contains reports whether the pixel (x, y) lies inside the rectangle.
func (b Bounds) Contains(x, y int) bool {
	return x >= b.MinX && x < b.MaxX && y >= b.MinY && y < b.MaxY
}

// Union returns the smallest rectangle that contains both b and other. The union
// of an empty rectangle with r is r.
func (b Bounds) Union(other Bounds) Bounds {
	if b.Empty() {
		return other
	}
	if other.Empty() {
		return b
	}
	return Bounds{
		MinX: min(b.MinX, other.MinX),
		MinY: min(b.MinY, other.MinY),
		MaxX: max(b.MaxX, other.MaxX),
		MaxY: max(b.MaxY, other.MaxY),
	}
}

// Intersect returns the overlap of b and other, which is empty if they do not
// overlap.
func (b Bounds) Intersect(other Bounds) Bounds {
	r := Bounds{
		MinX: max(b.MinX, other.MinX),
		MinY: max(b.MinY, other.MinY),
		MaxX: min(b.MaxX, other.MaxX),
		MaxY: min(b.MaxY, other.MaxY),
	}
	if r.Empty() {
		return Bounds{}
	}
	return r
}

// UnionBounds returns the smallest rectangle containing every rectangle in bs.
// It returns an empty Bounds if bs is empty.
func UnionBounds(bs []Bounds) Bounds {
	var u Bounds
	for _, b := range bs {
		u = u.Union(b)
	}
	return u
}

// WarpedBounds returns the integer bounding box that a width×height image
// occupies after being transformed by h. The four image corners are mapped and
// the box is expanded to whole pixels (floor of the minimum, ceil of the
// maximum).
func WarpedBounds(width, height int, h Homography) Bounds {
	corners := [4]PointF{
		{0, 0},
		{float64(width), 0},
		{0, float64(height)},
		{float64(width), float64(height)},
	}
	minX, minY := math.Inf(1), math.Inf(1)
	maxX, maxY := math.Inf(-1), math.Inf(-1)
	for _, c := range corners {
		p := h.Apply(c)
		minX = math.Min(minX, p.X)
		minY = math.Min(minY, p.Y)
		maxX = math.Max(maxX, p.X)
		maxY = math.Max(maxY, p.Y)
	}
	return Bounds{
		MinX: int(math.Floor(minX)),
		MinY: int(math.Floor(minY)),
		MaxX: int(math.Ceil(maxX)),
		MaxY: int(math.Ceil(maxY)),
	}
}

// PlaceImage copies src into dst so that the top-left corner of src lands at
// (offsetX, offsetY) in dst. Pixels of src that fall outside dst are clipped. It
// panics if the two images have different channel counts.
func PlaceImage(dst, src *cv.Mat, offsetX, offsetY int) {
	if dst.Channels != src.Channels {
		panic("stitch: PlaceImage channel mismatch")
	}
	ch := dst.Channels
	for y := 0; y < src.Rows; y++ {
		dy := y + offsetY
		if dy < 0 || dy >= dst.Rows {
			continue
		}
		for x := 0; x < src.Cols; x++ {
			dx := x + offsetX
			if dx < 0 || dx >= dst.Cols {
				continue
			}
			si := (y*src.Cols + x) * ch
			di := (dy*dst.Cols + dx) * ch
			copy(dst.Data[di:di+ch], src.Data[si:si+ch])
		}
	}
}
