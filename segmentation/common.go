package segmentation

// offset is a signed neighbour displacement in image coordinates.
type offset struct{ dx, dy int }

// neighbors4 is the 4-connected neighbourhood (edge neighbours).
var neighbors4 = []offset{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}

// neighbors8 is the 8-connected neighbourhood (edge and corner neighbours).
var neighbors8 = []offset{
	{1, 0}, {-1, 0}, {0, 1}, {0, -1},
	{1, 1}, {1, -1}, {-1, 1}, {-1, -1},
}

// clampU8 rounds v to the nearest integer and clamps it into [0, 255].
func clampU8(v float64) uint8 {
	v += 0.5
	if v <= 0 {
		return 0
	}
	if v >= 255 {
		return 255
	}
	return uint8(v)
}
