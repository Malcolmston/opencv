package bgsegm

// This file adds OpenCV-style getter/setter methods to the pre-existing
// subtractors. OpenCV exposes model parameters through get*/set* accessors
// (cv::BackgroundSubtractorMOG2::setHistory and friends) rather than public
// fields; these thin wrappers provide that familiar surface over the exported
// configuration fields, so callers can tune a model after construction without
// touching the fields directly. Every setter takes effect on the next Apply.

// --- BackgroundSubtractorMOG2 --------------------------------------------------

// SetHistory sets the nominal look-back length (History).
func (b *BackgroundSubtractorMOG2) SetHistory(history int) { b.History = history }

// GetHistory returns the nominal look-back length.
func (b *BackgroundSubtractorMOG2) GetHistory() int { return b.History }

// SetVarThreshold sets the squared-Mahalanobis match threshold (VarThreshold).
func (b *BackgroundSubtractorMOG2) SetVarThreshold(t float64) { b.VarThreshold = t }

// GetVarThreshold returns the squared-Mahalanobis match threshold.
func (b *BackgroundSubtractorMOG2) GetVarThreshold() float64 { return b.VarThreshold }

// SetDetectShadows enables or disables shadow classification.
func (b *BackgroundSubtractorMOG2) SetDetectShadows(on bool) { b.DetectShadows = on }

// GetDetectShadows reports whether shadow classification is enabled.
func (b *BackgroundSubtractorMOG2) GetDetectShadows() bool { return b.DetectShadows }

// SetShadowThreshold sets the darkest relative intensity still treated as a
// shadow.
func (b *BackgroundSubtractorMOG2) SetShadowThreshold(t float64) { b.ShadowThreshold = t }

// GetShadowThreshold returns the darkest relative intensity still treated as a
// shadow.
func (b *BackgroundSubtractorMOG2) GetShadowThreshold() float64 { return b.ShadowThreshold }

// SetNMixtures sets the maximum number of Gaussians kept per pixel. It has no
// effect once the model has been sized by the first Apply.
func (b *BackgroundSubtractorMOG2) SetNMixtures(n int) { b.NMixtures = n }

// GetNMixtures returns the maximum number of Gaussians kept per pixel.
func (b *BackgroundSubtractorMOG2) GetNMixtures() int { return b.NMixtures }

// SetBackgroundRatio sets the cumulative-weight fraction defining the background
// component set.
func (b *BackgroundSubtractorMOG2) SetBackgroundRatio(r float64) { b.BackgroundRatio = r }

// GetBackgroundRatio returns the cumulative-weight fraction defining the
// background component set.
func (b *BackgroundSubtractorMOG2) GetBackgroundRatio() float64 { return b.BackgroundRatio }

// --- BackgroundSubtractorKNN ---------------------------------------------------

// SetHistory sets the sample-bank refresh horizon (History).
func (b *BackgroundSubtractorKNN) SetHistory(history int) { b.History = history }

// GetHistory returns the sample-bank refresh horizon.
func (b *BackgroundSubtractorKNN) GetHistory() int { return b.History }

// SetDist2Threshold sets the squared intensity distance defining a neighbour.
func (b *BackgroundSubtractorKNN) SetDist2Threshold(t float64) { b.Dist2Threshold = t }

// GetDist2Threshold returns the squared intensity distance defining a neighbour.
func (b *BackgroundSubtractorKNN) GetDist2Threshold() float64 { return b.Dist2Threshold }

// SetKNNSamples sets the minimum neighbour count for a background classification.
func (b *BackgroundSubtractorKNN) SetKNNSamples(k int) { b.KNNSamples = k }

// GetKNNSamples returns the minimum neighbour count for a background
// classification.
func (b *BackgroundSubtractorKNN) GetKNNSamples() int { return b.KNNSamples }

// SetNSamples sets the number of samples stored per pixel. It has no effect once
// the model has been sized by the first Apply.
func (b *BackgroundSubtractorKNN) SetNSamples(n int) { b.NSamples = n }

// GetNSamples returns the number of samples stored per pixel.
func (b *BackgroundSubtractorKNN) GetNSamples() int { return b.NSamples }

// SetDetectShadows enables or disables shadow classification.
func (b *BackgroundSubtractorKNN) SetDetectShadows(on bool) { b.DetectShadows = on }

// GetDetectShadows reports whether shadow classification is enabled.
func (b *BackgroundSubtractorKNN) GetDetectShadows() bool { return b.DetectShadows }

// SetShadowThreshold sets the darkest relative intensity still treated as a
// shadow.
func (b *BackgroundSubtractorKNN) SetShadowThreshold(t float64) { b.ShadowThreshold = t }

// GetShadowThreshold returns the darkest relative intensity still treated as a
// shadow.
func (b *BackgroundSubtractorKNN) GetShadowThreshold() float64 { return b.ShadowThreshold }

// --- BackgroundSubtractorGMG ---------------------------------------------------

// SetNumInitFrames sets the length of the initial learning period.
func (b *BackgroundSubtractorGMG) SetNumInitFrames(n int) { b.NumInitFrames = n }

// GetNumInitFrames returns the length of the initial learning period.
func (b *BackgroundSubtractorGMG) GetNumInitFrames() int { return b.NumInitFrames }

// SetDecisionThreshold sets the foreground-probability decision threshold.
func (b *BackgroundSubtractorGMG) SetDecisionThreshold(t float64) { b.DecisionThreshold = t }

// GetDecisionThreshold returns the foreground-probability decision threshold.
func (b *BackgroundSubtractorGMG) GetDecisionThreshold() float64 { return b.DecisionThreshold }

// SetNumBins sets the number of intensity quantisation bins per pixel. It has no
// effect once the model has been sized by the first Apply.
func (b *BackgroundSubtractorGMG) SetNumBins(n int) { b.NumBins = n }

// GetNumBins returns the number of intensity quantisation bins per pixel.
func (b *BackgroundSubtractorGMG) GetNumBins() int { return b.NumBins }

// SetLearningRate sets the histogram aging rate.
func (b *BackgroundSubtractorGMG) SetLearningRate(r float64) { b.LearningRate = r }

// GetLearningRate returns the histogram aging rate.
func (b *BackgroundSubtractorGMG) GetLearningRate() float64 { return b.LearningRate }

// --- RunningAverage ------------------------------------------------------------

// SetAlpha sets the moving-average learning rate.
func (r *RunningAverage) SetAlpha(a float64) { r.Alpha = a }

// GetAlpha returns the moving-average learning rate.
func (r *RunningAverage) GetAlpha() float64 { return r.Alpha }

// SetThreshold sets the absolute intensity difference above which a pixel is
// foreground.
func (r *RunningAverage) SetThreshold(t float64) { r.Threshold = t }

// GetThreshold returns the foreground difference threshold.
func (r *RunningAverage) GetThreshold() float64 { return r.Threshold }
