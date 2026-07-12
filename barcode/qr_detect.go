package barcode

import (
	"math"
	"sort"

	cv "github.com/malcolmston/opencv"
)

// This file implements QR symbol localisation and sampling. Detection follows
// the classic approach: threshold the image to bi-level modules, scan rows for
// the finder pattern's distinctive 1:1:3:1:1 dark/light run-length signature,
// confirm each candidate with a vertical cross-check, cluster the hits into the
// three finder-pattern centres, deduce the symbol's orientation and version
// from their geometry, and sample every module on the finder-derived grid.
//
// The sampler uses an affine basis built from the three finder centres, so it
// tolerates translation, scaling, rotation and shear. Strong perspective is not
// corrected (that would use cv.GetPerspectiveTransform / cv.WarpPerspective and
// an alignment-pattern search); the images produced by [QREncode] are
// axis-aligned, so the affine model samples them exactly.

// finderCenter is a located finder pattern: its sub-pixel centre and the
// estimated module size (finder width / 7) in pixels.
type finderCenter struct {
	x, y   float64
	module float64
	n      int // supporting scanline hits, used for ranking and clustering
}

// toDarkGrid thresholds img to a boolean grid where true marks a dark module.
// It reuses the root package's colour conversion and Otsu threshold.
func toDarkGrid(img *cv.Mat) [][]bool {
	gray := img
	if img.Channels != 1 {
		gray = cv.CvtColor(img, cv.ColorRGB2Gray)
	}
	bin, _ := cv.Threshold(gray, 0, 255, cv.ThreshBinaryInv|cv.ThreshOtsu)
	h, w := bin.Rows, bin.Cols
	d := make([][]bool, h)
	for y := 0; y < h; y++ {
		row := make([]bool, w)
		for x := 0; x < w; x++ {
			row[x] = bin.Data[y*w+x] != 0
		}
		d[y] = row
	}
	return d
}

// finderRatioOK reports whether five consecutive run lengths approximate the
// 1:1:3:1:1 signature of a finder pattern crossing.
func finderRatioOK(sc [5]int) bool {
	total := 0
	for _, v := range sc {
		if v == 0 {
			return false
		}
		total += v
	}
	if total < 7 {
		return false
	}
	m := float64(total) / 7.0
	maxV := m * 0.5
	return math.Abs(float64(sc[0])-m) < maxV &&
		math.Abs(float64(sc[1])-m) < maxV &&
		math.Abs(float64(sc[2])-3*m) < 3*maxV &&
		math.Abs(float64(sc[3])-m) < maxV &&
		math.Abs(float64(sc[4])-m) < maxV
}

// centerFromEnd returns the centre column of the middle (3x) run given the run
// lengths and the position just past the last run.
func centerFromEnd(sc [5]int, end int) float64 {
	return float64(end) - float64(sc[4]) - float64(sc[3]) - float64(sc[2])/2
}

// findFinderCenters scans every row for the finder signature, cross-checks each
// candidate vertically, and clusters the surviving hits into finder centres.
func findFinderCenters(dark [][]bool) []finderCenter {
	h := len(dark)
	if h == 0 {
		return nil
	}
	w := len(dark[0])
	var raw []finderCenter
	for y := 0; y < h; y++ {
		var sc [5]int
		state := 0
		for x := 0; x < w; x++ {
			if dark[y][x] {
				if state&1 == 1 {
					state++
				}
				sc[state]++
			} else {
				if state&1 == 0 {
					if state == 4 {
						if finderRatioOK(sc) {
							cx := centerFromEnd(sc, x)
							if cy, module, ok := crossCheckVertical(dark, y, int(cx+0.5), sc[2], sc[0]+sc[1]+sc[2]+sc[3]+sc[4]); ok {
								raw = append(raw, finderCenter{x: cx, y: cy, module: module, n: 1})
							}
							sc = [5]int{sc[2], sc[3], sc[4], 1, 0}
							state = 3
							continue
						}
						sc = [5]int{sc[2], sc[3], sc[4], 1, 0}
						state = 3
						continue
					}
					state++
					sc[state]++
				} else {
					sc[state]++
				}
			}
		}
	}
	return clusterCenters(raw)
}

// crossCheckVertical confirms a finder candidate by walking the 1:1:3:1:1
// pattern vertically through the centre column, returning the refined centre row
// and module size.
func crossCheckVertical(dark [][]bool, startRow, centerCol, centerCount, total int) (float64, float64, bool) {
	h := len(dark)
	if centerCol < 0 || centerCol >= len(dark[0]) {
		return 0, 0, false
	}
	maxCount := centerCount
	var sc [5]int
	row := startRow
	for row >= 0 && dark[row][centerCol] {
		sc[2]++
		row--
	}
	if row < 0 {
		return 0, 0, false
	}
	for row >= 0 && !dark[row][centerCol] && sc[1] <= maxCount {
		sc[1]++
		row--
	}
	if row < 0 || sc[1] > maxCount {
		return 0, 0, false
	}
	for row >= 0 && dark[row][centerCol] && sc[0] <= maxCount {
		sc[0]++
		row--
	}
	if sc[0] > maxCount {
		return 0, 0, false
	}
	row = startRow + 1
	for row < h && dark[row][centerCol] {
		sc[2]++
		row++
	}
	if row == h {
		return 0, 0, false
	}
	for row < h && !dark[row][centerCol] && sc[3] <= maxCount {
		sc[3]++
		row++
	}
	if row == h || sc[3] > maxCount {
		return 0, 0, false
	}
	for row < h && dark[row][centerCol] && sc[4] <= maxCount {
		sc[4]++
		row++
	}
	if sc[4] > maxCount {
		return 0, 0, false
	}
	stateTotal := sc[0] + sc[1] + sc[2] + sc[3] + sc[4]
	if 5*abs(stateTotal-total) >= 2*total {
		return 0, 0, false
	}
	if !finderRatioOK(sc) {
		return 0, 0, false
	}
	centerY := centerFromEnd(sc, row)
	return centerY, float64(stateTotal) / 7.0, true
}

// clusterCenters merges nearby finder hits (each finder yields many rows of
// hits) into averaged centres.
func clusterCenters(raw []finderCenter) []finderCenter {
	var cl []finderCenter
	for _, c := range raw {
		merged := false
		for i := range cl {
			tol := cl[i].module * 2
			if math.Abs(c.x-cl[i].x) < tol && math.Abs(c.y-cl[i].y) < tol {
				n := float64(cl[i].n)
				cl[i].x = (cl[i].x*n + c.x) / (n + 1)
				cl[i].y = (cl[i].y*n + c.y) / (n + 1)
				cl[i].module = (cl[i].module*n + c.module) / (n + 1)
				cl[i].n++
				merged = true
				break
			}
		}
		if !merged {
			cl = append(cl, c)
		}
	}
	return cl
}

// strongCenters keeps only well-supported clusters (finder patterns produce
// many scanline hits; stray matches produce few), sorted by support descending.
func strongCenters(centers []finderCenter) []finderCenter {
	sort.Slice(centers, func(i, j int) bool { return centers[i].n > centers[j].n })
	if len(centers) == 0 {
		return centers
	}
	maxN := centers[0].n
	out := centers[:0:0]
	for _, c := range centers {
		if float64(c.n) >= 0.4*float64(maxN) {
			out = append(out, c)
		}
	}
	return out
}

// FindFinderPatterns locates the QR finder patterns in img and returns their
// centre points (in pixels), strongest first. For a clean single QR symbol this
// yields exactly three points. It is exported so callers can verify
// localisation independently of decoding.
func FindFinderPatterns(img *cv.Mat) []cv.Point {
	if img == nil || img.Empty() {
		return nil
	}
	centers := strongCenters(findFinderCenters(toDarkGrid(img)))
	out := make([]cv.Point, 0, len(centers))
	for _, c := range centers {
		out = append(out, cv.Point{X: int(c.x + 0.5), Y: int(c.y + 0.5)})
	}
	return out
}

// orientFinders labels the three centres as top-left, top-right and bottom-left
// by finding the right-angle vertex (opposite the longest side) and fixing the
// handedness with a cross product.
func orientFinders(c [3]finderCenter) (tl, tr, bl finderCenter) {
	d01 := distSq(c[0], c[1])
	d12 := distSq(c[1], c[2])
	d02 := distSq(c[0], c[2])
	// The vertex not on the hypotenuse (longest side) is the top-left corner.
	var a, b int
	switch {
	case d12 >= d01 && d12 >= d02:
		tl, a, b = c[0], 1, 2
	case d02 >= d01 && d02 >= d12:
		tl, a, b = c[1], 0, 2
	default:
		tl, a, b = c[2], 0, 1
	}
	pa, pb := c[a], c[b]
	cross := (pa.x-tl.x)*(pb.y-tl.y) - (pa.y-tl.y)*(pb.x-tl.x)
	if cross >= 0 {
		return tl, pa, pb
	}
	return tl, pb, pa
}

func distSq(a, b finderCenter) float64 {
	dx, dy := a.x-b.x, a.y-b.y
	return dx*dx + dy*dy
}

// sampleGrid samples a dim x dim module grid from dark using the affine basis
// derived from the three finder centres. Each centre lies at module index 3
// (the middle of the 7-module-wide finder), so the centres are dim-7 modules
// apart along each edge.
func sampleGrid(dark [][]bool, tl, tr, bl finderCenter, dim int) [][]bool {
	h, w := len(dark), len(dark[0])
	span := float64(dim - 7)
	axCol := [2]float64{(tr.x - tl.x) / span, (tr.y - tl.y) / span}
	axRow := [2]float64{(bl.x - tl.x) / span, (bl.y - tl.y) / span}
	grid := make([][]bool, dim)
	for r := 0; r < dim; r++ {
		grid[r] = make([]bool, dim)
		for c := 0; c < dim; c++ {
			fc := float64(c) - 3
			fr := float64(r) - 3
			px := tl.x + fc*axCol[0] + fr*axRow[0]
			py := tl.y + fc*axCol[1] + fr*axRow[1]
			xi, yi := int(px+0.5), int(py+0.5)
			if xi >= 0 && xi < w && yi >= 0 && yi < h {
				grid[r][c] = dark[yi][xi]
			}
		}
	}
	return grid
}

// snapVersion converts an estimated module dimension to the nearest supported
// symbol version (1-4), returning false if it is out of range.
func snapVersion(dimEst float64) (version, dim int, ok bool) {
	best, bestErr := 0, math.MaxFloat64
	for v := 1; v <= 4; v++ {
		d := float64(v*4 + 17)
		if e := math.Abs(d - dimEst); e < bestErr {
			bestErr = e
			best = v
		}
	}
	if best == 0 || bestErr > 2 {
		return 0, 0, false
	}
	return best, best*4 + 17, true
}

// QRDetectAndDecode locates a QR Code symbol in img, samples its modules and
// decodes the byte-mode payload, returning the text and true on success. It
// returns ("", false) if no decodable symbol is found. The decoder reads the
// mask from the format information and applies Reed-Solomon error correction,
// so it recovers the message even when a few modules are misread. It handles the
// byte-mode, level-L, version 1-4 symbols produced by [QREncode] (including
// axis-aligned rotations); see the package documentation for scope.
func QRDetectAndDecode(img *cv.Mat) (string, bool) {
	if img == nil || img.Empty() {
		return "", false
	}
	dark := toDarkGrid(img)
	centers := strongCenters(findFinderCenters(dark))
	if len(centers) < 3 {
		return "", false
	}
	var three [3]finderCenter
	copy(three[:], centers[:3])
	tl, tr, bl := orientFinders(three)

	module := (tl.module + tr.module + bl.module) / 3
	if module <= 0 {
		return "", false
	}
	distTR := math.Hypot(tr.x-tl.x, tr.y-tl.y)
	distBL := math.Hypot(bl.x-tl.x, bl.y-tl.y)
	dimEst := (distTR+distBL)/(2*module) + 7
	version, dim, ok := snapVersion(dimEst)
	if !ok {
		return "", false
	}
	grid := sampleGrid(dark, tl, tr, bl, dim)
	return decodeMatrix(grid, version)
}
