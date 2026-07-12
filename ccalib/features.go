package ccalib

import (
	"math"
	"sort"

	cv "github.com/malcolmston/opencv"
)

// This file implements the self-contained feature-detection and matching core
// shared by the custom-pattern and random-pattern calibration targets. It uses
// only the root cv package (thresholding + connected components) and the
// standard library. Random-dot targets are matched by a rotation-, scale- and
// translation-invariant geometric descriptor rather than by pixel-patch
// appearance, so the pipeline tolerates the perspective warp between the
// reference target and a captured view.

// blob is a detected dot: its sub-pixel centre (X, Y) and approximate radius.
type blob struct {
	X, Y float64
	R    float64
}

// descriptorNeighbors is how many nearest blobs form a blob's geometric
// descriptor.
const descriptorNeighbors = 6

// toGray returns a single-channel view of img (luminance for colour input).
func toGray(img *cv.Mat) *cv.Mat {
	if img.Channels == 1 {
		return img
	}
	g := cv.NewMat(img.Rows, img.Cols, 1)
	for y := 0; y < img.Rows; y++ {
		for x := 0; x < img.Cols; x++ {
			px := img.AtPixel(y, x)
			var lum float64
			if len(px) >= 3 {
				lum = 0.299*float64(px[0]) + 0.587*float64(px[1]) + 0.114*float64(px[2])
			} else {
				lum = float64(px[0])
			}
			g.Set(y, x, 0, uint8(lum+0.5))
		}
	}
	return g
}

// detectBlobs finds the bright dots of a random-dot target: it binarises the
// image with Otsu's threshold, labels the connected foreground components and
// returns their centroids and radii, filtered by area. minArea and maxArea bound
// the accepted component size in pixels; a non-positive maxArea disables the
// upper bound.
func detectBlobs(img *cv.Mat, minArea, maxArea int) []blob {
	gray := toGray(img)
	binary, _ := cv.Threshold(gray, 0, 255, cv.ThreshBinary|cv.ThreshOtsu)
	_, count, stats := cv.ConnectedComponentsWithStats(binary, cv.Connectivity8)
	var out []blob
	for i := 0; i < count; i++ {
		s := stats[i]
		if s.Label == 0 {
			continue
		}
		// Background label is the largest bright region only if the target is
		// inverted; guard by skipping components touching every border via area.
		if s.Area < minArea {
			continue
		}
		if maxArea > 0 && s.Area > maxArea {
			continue
		}
		out = append(out, blob{
			X: s.CentroidX,
			Y: s.CentroidY,
			R: math.Sqrt(float64(s.Area) / math.Pi),
		})
	}
	return out
}

// descriptor is the invariant signature of one blob relative to its neighbours.
type descriptor struct {
	vec   []float64 // length 2*descriptorNeighbors: normalised distances then relative angles
	valid bool
}

// buildDescriptors computes a geometric descriptor for every blob. The
// descriptor lists the blob's nearest neighbours ordered by angle relative to
// the closest neighbour (rotation invariance), with distances normalised by
// their mean (scale invariance). Blobs with too few neighbours are marked
// invalid.
func buildDescriptors(blobs []blob) []descriptor {
	n := len(blobs)
	descs := make([]descriptor, n)
	for i := range blobs {
		type nb struct {
			d, ang float64
		}
		neigh := make([]nb, 0, n-1)
		for j := range blobs {
			if j == i {
				continue
			}
			dx := blobs[j].X - blobs[i].X
			dy := blobs[j].Y - blobs[i].Y
			neigh = append(neigh, nb{d: math.Hypot(dx, dy), ang: math.Atan2(dy, dx)})
		}
		if len(neigh) < descriptorNeighbors {
			continue
		}
		sort.Slice(neigh, func(a, b int) bool { return neigh[a].d < neigh[b].d })
		neigh = neigh[:descriptorNeighbors]
		ref := neigh[0].ang
		var meanD float64
		for _, e := range neigh {
			meanD += e.d
		}
		meanD /= float64(len(neigh))
		if meanD < 1e-9 {
			continue
		}
		type entry struct{ d, rel float64 }
		entries := make([]entry, len(neigh))
		for k, e := range neigh {
			rel := math.Mod(e.ang-ref+2*math.Pi, 2*math.Pi)
			entries[k] = entry{d: e.d / meanD, rel: rel}
		}
		sort.Slice(entries, func(a, b int) bool { return entries[a].rel < entries[b].rel })
		vec := make([]float64, 2*len(entries))
		for k, e := range entries {
			vec[k] = e.d
			vec[len(entries)+k] = e.rel
		}
		descs[i] = descriptor{vec: vec, valid: true}
	}
	return descs
}

// descDistance returns the squared Euclidean distance between two descriptors,
// weighting the angular half so a full turn is comparable to the distance half.
func descDistance(a, b []float64) float64 {
	half := len(a) / 2
	var s float64
	for i := 0; i < half; i++ {
		d := a[i] - b[i]
		s += d * d
	}
	for i := half; i < len(a); i++ {
		d := a[i] - b[i]
		// Wrap angular differences into (-π, π].
		for d > math.Pi {
			d -= 2 * math.Pi
		}
		for d < -math.Pi {
			d += 2 * math.Pi
		}
		s += d * d
	}
	return s
}

// match pairs a reference blob index with a query blob index.
type match struct {
	ref, query int
}

// matchBlobs returns the mutual, ratio-tested descriptor matches between a
// reference and a query blob set. A query blob matches a reference blob only
// when the reference is its best match by a clear margin (Lowe's ratio test) and
// the query is likewise the reference's best match.
func matchBlobs(refDesc, queryDesc []descriptor) []match {
	best := make([]int, len(queryDesc))
	for qi := range queryDesc {
		best[qi] = -1
		if !queryDesc[qi].valid {
			continue
		}
		b1, b2 := math.Inf(1), math.Inf(1)
		bi := -1
		for ri := range refDesc {
			if !refDesc[ri].valid {
				continue
			}
			d := descDistance(queryDesc[qi].vec, refDesc[ri].vec)
			if d < b1 {
				b2 = b1
				b1 = d
				bi = ri
			} else if d < b2 {
				b2 = d
			}
		}
		if bi >= 0 && b1 < 0.7*b2 {
			best[qi] = bi
		}
	}
	// Mutual check: the reference's own best query must be this query.
	refBest := make([]int, len(refDesc))
	for ri := range refDesc {
		refBest[ri] = -1
		if !refDesc[ri].valid {
			continue
		}
		b1 := math.Inf(1)
		bi := -1
		for qi := range queryDesc {
			if !queryDesc[qi].valid {
				continue
			}
			d := descDistance(refDesc[ri].vec, queryDesc[qi].vec)
			if d < b1 {
				b1 = d
				bi = qi
			}
		}
		refBest[ri] = bi
	}
	var matches []match
	for qi, ri := range best {
		if ri >= 0 && refBest[ri] == qi {
			matches = append(matches, match{ref: ri, query: qi})
		}
	}
	return matches
}

// normalizePoints applies an isotropic (Hartley) normalisation to a point set,
// returning the transformed points and the 3×3 similarity transform T mapping
// the originals to them.
func normalizePoints(pts [][2]float64) ([][2]float64, [3][3]float64) {
	var cx, cy float64
	for _, p := range pts {
		cx += p[0]
		cy += p[1]
	}
	n := float64(len(pts))
	cx /= n
	cy /= n
	var meanDist float64
	for _, p := range pts {
		meanDist += math.Hypot(p[0]-cx, p[1]-cy)
	}
	meanDist /= n
	scale := 1.0
	if meanDist > 1e-12 {
		scale = math.Sqrt2 / meanDist
	}
	T := [3][3]float64{{scale, 0, -scale * cx}, {0, scale, -scale * cy}, {0, 0, 1}}
	out := make([][2]float64, len(pts))
	for i, p := range pts {
		out[i] = [2]float64{scale * (p[0] - cx), scale * (p[1] - cy)}
	}
	return out, T
}

// dltHomography estimates the homography H mapping src points to dst points via
// the normalised Direct Linear Transform. At least four correspondences are
// required. ok is false when the configuration is degenerate.
func dltHomography(src, dst [][2]float64) ([3][3]float64, bool) {
	if len(src) < 4 || len(src) != len(dst) {
		return [3][3]float64{}, false
	}
	ns, ts := normalizePoints(src)
	nd, td := normalizePoints(dst)
	var rows [][]float64
	for i := range ns {
		x, y := ns[i][0], ns[i][1]
		u, v := nd[i][0], nd[i][1]
		rows = append(rows, []float64{-x, -y, -1, 0, 0, 0, u * x, u * y, u})
		rows = append(rows, []float64{0, 0, 0, -x, -y, -1, v * x, v * y, v})
	}
	h := nullspaceVec(rows, 9)
	Hn := [3][3]float64{
		{h[0], h[1], h[2]},
		{h[3], h[4], h[5]},
		{h[6], h[7], h[8]},
	}
	// Denormalise: H = Td^{-1} · Hn · Ts.
	tdInv, ok := inv3(td)
	if !ok {
		return [3][3]float64{}, false
	}
	H := mul3(mul3(tdInv, Hn), ts)
	if math.Abs(H[2][2]) > 1e-12 {
		inv := 1 / H[2][2]
		for i := 0; i < 3; i++ {
			for j := 0; j < 3; j++ {
				H[i][j] *= inv
			}
		}
	}
	return H, true
}

// applyHomography maps a point through a 3×3 homography.
func applyHomography(H [3][3]float64, p [2]float64) [2]float64 {
	w := H[2][0]*p[0] + H[2][1]*p[1] + H[2][2]
	if math.Abs(w) < 1e-15 {
		w = 1e-15
	}
	return [2]float64{
		(H[0][0]*p[0] + H[0][1]*p[1] + H[0][2]) / w,
		(H[1][0]*p[0] + H[1][1]*p[1] + H[1][2]) / w,
	}
}

// filterMatchesByHomography fits a homography to the candidate matches, keeps
// those whose reprojection error is within tol pixels, and refits once. It
// returns the surviving matches and the refined homography (reference → query).
// ok is false when fewer than four consistent matches remain.
func filterMatchesByHomography(refBlobs, queryBlobs []blob, matches []match, tol float64) ([]match, [3][3]float64, bool) {
	if len(matches) < 4 {
		return nil, [3][3]float64{}, false
	}
	src := make([][2]float64, len(matches))
	dst := make([][2]float64, len(matches))
	for i, m := range matches {
		src[i] = [2]float64{refBlobs[m.ref].X, refBlobs[m.ref].Y}
		dst[i] = [2]float64{queryBlobs[m.query].X, queryBlobs[m.query].Y}
	}
	H, ok := dltHomography(src, dst)
	if !ok {
		return nil, [3][3]float64{}, false
	}
	var inliers []match
	var isrc, idst [][2]float64
	for i, m := range matches {
		pred := applyHomography(H, src[i])
		if math.Hypot(pred[0]-dst[i][0], pred[1]-dst[i][1]) <= tol {
			inliers = append(inliers, m)
			isrc = append(isrc, src[i])
			idst = append(idst, dst[i])
		}
	}
	if len(inliers) < 4 {
		return nil, [3][3]float64{}, false
	}
	if H2, ok2 := dltHomography(isrc, idst); ok2 {
		H = H2
	}
	return inliers, H, true
}
