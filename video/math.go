package video

import "math"

// floor is a thin alias for math.Floor kept local for terse call sites.
func floor(x float64) float64 { return math.Floor(x) }

// abs is a thin alias for math.Abs.
func abs(x float64) float64 { return math.Abs(x) }

// sqrt is a thin alias for math.Sqrt.
func sqrt(x float64) float64 { return math.Sqrt(x) }
