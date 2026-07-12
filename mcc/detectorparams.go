package mcc

import "errors"

// DetectorParameters bundles the tuning knobs of the classical
// [CCheckerDetector] into a single, validatable configuration object, mirroring
// OpenCV's mcc DetectorParameters. Obtain sensible defaults from
// [NewDetectorParameters], adjust the fields, then build a detector with
// [DetectorParameters.NewDetector].
type DetectorParameters struct {
	// MinPatchAreaFrac and MaxPatchAreaFrac bound a patch-candidate contour's
	// area as a fraction of the whole image.
	MinPatchAreaFrac float64
	MaxPatchAreaFrac float64
	// ApproxEpsilonFrac is the Douglas–Peucker tolerance for reducing a contour
	// to a quadrilateral, as a fraction of the contour perimeter.
	ApproxEpsilonFrac float64
}

// NewDetectorParameters returns a DetectorParameters populated with the same
// defaults [NewCCheckerDetector] uses: they suit charts that occupy a good
// fraction of a reasonably-exposed frame.
func NewDetectorParameters() *DetectorParameters {
	return &DetectorParameters{
		MinPatchAreaFrac:  0.0005,
		MaxPatchAreaFrac:  0.2,
		ApproxEpsilonFrac: 0.08,
	}
}

// Validate reports whether the parameters are internally consistent: positive
// fractions, a minimum area below the maximum, and both area fractions within
// (0,1]. It returns a descriptive error otherwise.
func (p *DetectorParameters) Validate() error {
	if p.MinPatchAreaFrac <= 0 || p.MinPatchAreaFrac > 1 {
		return errors.New("mcc: MinPatchAreaFrac must be in (0,1]")
	}
	if p.MaxPatchAreaFrac <= 0 || p.MaxPatchAreaFrac > 1 {
		return errors.New("mcc: MaxPatchAreaFrac must be in (0,1]")
	}
	if p.MinPatchAreaFrac >= p.MaxPatchAreaFrac {
		return errors.New("mcc: MinPatchAreaFrac must be less than MaxPatchAreaFrac")
	}
	if p.ApproxEpsilonFrac <= 0 || p.ApproxEpsilonFrac >= 1 {
		return errors.New("mcc: ApproxEpsilonFrac must be in (0,1)")
	}
	return nil
}

// NewDetector builds a [CCheckerDetector] for the given chart type using these
// parameters. The returned detector is independent; further edits to the
// parameters do not affect it.
func (p *DetectorParameters) NewDetector(t CheckerType) *CCheckerDetector {
	return &CCheckerDetector{
		Type:              t,
		MinPatchAreaFrac:  p.MinPatchAreaFrac,
		MaxPatchAreaFrac:  p.MaxPatchAreaFrac,
		ApproxEpsilonFrac: p.ApproxEpsilonFrac,
	}
}
