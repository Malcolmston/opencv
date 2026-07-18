package shapefit

import (
	"math"
	"sort"

	cv "github.com/malcolmston/opencv"
)

// HoughMatch is a candidate placement of a template found by the generalized
// Hough transform.
type HoughMatch struct {
	// Location is the detected position of the template reference point.
	Location cv.Point2f
	// Votes is the accumulator support for the placement.
	Votes int
}

// GeneralizedHough implements Ballard's generalized Hough transform for
// detecting instances of an arbitrary shape described by a template point set.
// It builds an R-table that maps quantized local edge orientation to the set of
// displacement vectors from each edge point to a chosen reference point. At
// detection time each query edge point votes, for every displacement stored
// under its orientation bin, for a candidate reference location.
//
// This implementation detects translated instances of the template at the
// original scale and orientation. It does not search over scale or rotation.
type GeneralizedHough struct {
	reference cv.Point2f
	bins      int
	neighbors int
	// rtable[bin] holds the displacement vectors (reference - edgePoint) for
	// template points whose orientation falls in that bin.
	rtable [][]cv.Point2f
}

// NewGeneralizedHough builds a generalized Hough model from a template edge
// point set and a reference point (commonly the template centroid). Local edge
// orientation at each template point is estimated from its neighbors and
// quantized into orientationBins bins spanning [0, π). neighbors controls the
// neighborhood size used for orientation estimation; if it is below 1 a default
// of 4 is used. It returns nil if the template is empty or orientationBins is
// not positive.
func NewGeneralizedHough(template []cv.Point2f, reference cv.Point2f, orientationBins, neighbors int) *GeneralizedHough {
	if len(template) == 0 || orientationBins < 1 {
		return nil
	}
	if neighbors < 1 {
		neighbors = 4
	}
	g := &GeneralizedHough{
		reference: reference,
		bins:      orientationBins,
		neighbors: neighbors,
		rtable:    make([][]cv.Point2f, orientationBins),
	}
	orient := EstimateOrientations(template, neighbors)
	for i, p := range template {
		b := g.binOf(orient[i])
		g.rtable[b] = append(g.rtable[b], cv.Point2f{
			X: reference.X - p.X,
			Y: reference.Y - p.Y,
		})
	}
	return g
}

// binOf maps an orientation in [0, π) to an R-table bin index.
func (g *GeneralizedHough) binOf(theta float64) int {
	theta = math.Mod(theta, math.Pi)
	if theta < 0 {
		theta += math.Pi
	}
	b := int(theta / math.Pi * float64(g.bins))
	if b >= g.bins {
		b = g.bins - 1
	}
	if b < 0 {
		b = 0
	}
	return b
}

// Reference returns the reference point the model was built with.
func (g *GeneralizedHough) Reference() cv.Point2f { return g.reference }

// Bins returns the number of orientation bins in the R-table.
func (g *GeneralizedHough) Bins() int { return g.bins }

// Detect locates instances of the template in a query edge point set spanning a
// width×height image. It accumulates reference-point votes and returns every
// accumulator peak with at least threshold votes, sorted by descending votes.
// The location resolution is one pixel.
func (g *GeneralizedHough) Detect(query []cv.Point2f, width, height, threshold int) []HoughMatch {
	if g == nil || width < 1 || height < 1 || len(query) == 0 {
		return nil
	}
	acc := make([]int, width*height)
	orient := EstimateOrientations(query, g.neighbors)
	for i, p := range query {
		b := g.binOf(orient[i])
		for _, off := range g.rtable[b] {
			cx := int(math.Round(p.X + off.X))
			cy := int(math.Round(p.Y + off.Y))
			if cx >= 0 && cx < width && cy >= 0 && cy < height {
				acc[cy*width+cx]++
			}
		}
	}
	var out []HoughMatch
	for cy := 0; cy < height; cy++ {
		for cx := 0; cx < width; cx++ {
			v := acc[cy*width+cx]
			if v < threshold {
				continue
			}
			if !shapefitIsPeakGrid(acc, cx, cy, width, height) {
				continue
			}
			out = append(out, HoughMatch{
				Location: cv.Point2f{X: float64(cx), Y: float64(cy)},
				Votes:    v,
			})
		}
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].Votes > out[j].Votes })
	return out
}
