package imgprocx

import (
	"math"
	"sort"

	cv "github.com/malcolmston/opencv"
)

// HoughLine3 is one line found by [HoughLinesPointSet]: Votes point counts, and
// the line is in Hesse normal form x·cos(Theta) + y·sin(Theta) = Rho (Rho in the
// same units as the point coordinates, Theta in radians).
type HoughLine3 struct {
	Votes int
	Rho   float64
	Theta float64
}

// HoughLinesPointSet finds the lines best supported by an arbitrary set of 2-D
// points using the standard Hough transform, mirroring cv2.HoughLinesPointSet.
// Rather than scanning an image it votes directly from the given points, which
// is the natural form when the candidate points are sparse (for example feature
// locations).
//
// The accumulator spans rho in [minRho, maxRho] quantised by rhoStep and theta
// in [minTheta, maxTheta) quantised by thetaStep. Each point casts a vote for
// every theta bin at the rho bin nearest x·cosθ + y·sinθ (votes whose rho falls
// outside the range are discarded). Every accumulator cell with at least
// threshold votes is returned as a [HoughLine3], sorted by descending votes
// (ties broken by rho then theta for determinism), truncated to at most linesMax
// entries. It panics on non-positive steps or an inverted range.
func HoughLinesPointSet(points []cv.Point, linesMax, threshold int,
	minRho, maxRho, rhoStep, minTheta, maxTheta, thetaStep float64) []HoughLine3 {
	if rhoStep <= 0 || thetaStep <= 0 {
		panic("imgprocx: HoughLinesPointSet requires positive rhoStep and thetaStep")
	}
	if maxRho < minRho || maxTheta < minTheta {
		panic("imgprocx: HoughLinesPointSet requires max >= min for rho and theta")
	}
	numRho := int(math.Round((maxRho-minRho)/rhoStep)) + 1
	numTheta := int(math.Round((maxTheta - minTheta) / thetaStep))
	if numTheta <= 0 {
		numTheta = 1
	}
	// Precompute the sinusoid basis for each theta bin.
	cosT := make([]float64, numTheta)
	sinT := make([]float64, numTheta)
	for t := 0; t < numTheta; t++ {
		ang := minTheta + float64(t)*thetaStep
		cosT[t] = math.Cos(ang)
		sinT[t] = math.Sin(ang)
	}
	acc := make([]int, numTheta*numRho)
	for _, p := range points {
		px, py := float64(p.X), float64(p.Y)
		for t := 0; t < numTheta; t++ {
			rho := px*cosT[t] + py*sinT[t]
			r := int(math.Round((rho - minRho) / rhoStep))
			if r < 0 || r >= numRho {
				continue
			}
			acc[t*numRho+r]++
		}
	}
	var out []HoughLine3
	for t := 0; t < numTheta; t++ {
		for r := 0; r < numRho; r++ {
			v := acc[t*numRho+r]
			if v >= threshold {
				out = append(out, HoughLine3{
					Votes: v,
					Rho:   minRho + float64(r)*rhoStep,
					Theta: minTheta + float64(t)*thetaStep,
				})
			}
		}
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Votes != out[j].Votes {
			return out[i].Votes > out[j].Votes
		}
		if out[i].Rho != out[j].Rho {
			return out[i].Rho < out[j].Rho
		}
		return out[i].Theta < out[j].Theta
	})
	if linesMax >= 0 && len(out) > linesMax {
		out = out[:linesMax]
	}
	return out
}
