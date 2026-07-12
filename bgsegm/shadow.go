package bgsegm

// ShadowParams is the shadow-detection configuration shared, by embedding, among
// the subtractors that can recognise the moving shadow of a background object
// (currently [BackgroundSubtractorMOG], [BackgroundSubtractorCNT],
// [BackgroundSubtractorLSBP] and [BackgroundSubtractorGSOC]). Its exported
// fields are promoted onto the embedding subtractor, so they can be read and
// written directly (for example sub.DetectShadows = true) or through the
// OpenCV-style getter/setter methods.
//
// A pixel is reported as a shadow — emitting [ShadowValue] (or the configured
// override) instead of [ForegroundValue] — when it is darker than the modelled
// background yet no darker than ShadowThreshold times the background intensity,
// i.e. it looks like the same surface under reduced illumination rather than a
// genuinely different object.
type ShadowParams struct {
	// DetectShadows enables the shadow classification. When false a darkened
	// background pixel is reported as ordinary foreground.
	DetectShadows bool
	// ShadowValue is the mask sample emitted for a detected shadow. It defaults
	// to the package constant [ShadowValue] (127); set it to another value to
	// distinguish a model's shadows in a combined mask.
	ShadowValue uint8
	// ShadowThreshold is the darkest relative intensity (observation ÷
	// background) still accepted as a shadow. The default 0.5 accepts pixels down
	// to half the background brightness; a smaller value accepts deeper shadows.
	ShadowThreshold float64
}

// defaultShadowParams returns the shadow configuration used by every new
// subtractor: detection off, the standard [ShadowValue] sample and a 0.5
// relative-intensity floor.
func defaultShadowParams() ShadowParams {
	return ShadowParams{
		DetectShadows:   false,
		ShadowValue:     ShadowValue,
		ShadowThreshold: 0.5,
	}
}

// SetDetectShadows enables or disables shadow classification.
func (s *ShadowParams) SetDetectShadows(on bool) { s.DetectShadows = on }

// GetDetectShadows reports whether shadow classification is enabled.
func (s *ShadowParams) GetDetectShadows() bool { return s.DetectShadows }

// SetShadowValue sets the mask sample emitted for detected shadows.
func (s *ShadowParams) SetShadowValue(v uint8) { s.ShadowValue = v }

// GetShadowValue returns the mask sample emitted for detected shadows.
func (s *ShadowParams) GetShadowValue() uint8 { return s.shadowSample() }

// SetShadowThreshold sets the darkest relative intensity still treated as a
// shadow. Values outside (0,1] are accepted but disable useful shadow detection.
func (s *ShadowParams) SetShadowThreshold(t float64) { s.ShadowThreshold = t }

// GetShadowThreshold returns the darkest relative intensity still treated as a
// shadow.
func (s *ShadowParams) GetShadowThreshold() float64 { return s.ShadowThreshold }

// shadowSample returns the configured shadow mask sample, falling back to the
// package [ShadowValue] when the field was left at its zero value.
func (s *ShadowParams) shadowSample() uint8 {
	if s.ShadowValue == 0 {
		return ShadowValue
	}
	return s.ShadowValue
}

// isShadowOf reports whether observation v looks like a darkened version of the
// background reference ref: dimmer than ref but no darker than
// ShadowThreshold·ref. It always returns false when detection is disabled or the
// reference is non-positive.
func (s *ShadowParams) isShadowOf(v, ref float64) bool {
	if !s.DetectShadows || ref <= 0 {
		return false
	}
	return v <= ref && v >= s.ShadowThreshold*ref
}
