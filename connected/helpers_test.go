package connected

import cv "github.com/malcolmston/opencv"

// matFromRows builds a single-channel binary Mat from ASCII rows. A '#', '1' or
// 'X' rune is foreground (255); any other rune is background (0). All rows must
// share the same length.
func matFromRows(rows []string) *cv.Mat {
	h := len(rows)
	w := len(rows[0])
	m := cv.NewMat(h, w, 1)
	for y, r := range rows {
		if len(r) != w {
			panic("matFromRows: ragged rows")
		}
		for x, c := range r {
			if c == '#' || c == '1' || c == 'X' {
				m.Data[y*w+x] = 255
			}
		}
	}
	return m
}

// countValue returns the number of samples in m equal to v.
func countValue(m *cv.Mat, v uint8) int {
	n := 0
	for _, s := range m.Data {
		if s == v {
			n++
		}
	}
	return n
}

// pointSet returns a set-membership map keyed by "x,y" for the given points.
func pointSet(pts []cv.Point) map[[2]int]bool {
	s := make(map[[2]int]bool, len(pts))
	for _, p := range pts {
		s[[2]int{p.X, p.Y}] = true
	}
	return s
}
