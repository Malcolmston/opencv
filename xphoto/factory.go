package xphoto

// This file adds OpenCV-style factory functions and get/set accessors for the
// white-balance types declared in whitebalance.go. OpenCV exposes the white
// balancers through create*WB() factories and getX/setX property accessors
// rather than public fields; these wrappers provide the same call shape on top
// of the field-based Go types, so code ported from the C++ API reads naturally
// while the plain-struct construction (NewSimpleWB, direct field assignment)
// keeps working unchanged.

// CreateSimpleWB returns a new [SimpleWB] with OpenCV's default parameters. It
// is the direct analogue of cv::xphoto::createSimpleWB and is equivalent to
// [NewSimpleWB].
func CreateSimpleWB() *SimpleWB { return NewSimpleWB() }

// CreateGrayworldWB returns a new [GrayworldWB] with OpenCV's default
// parameters, mirroring cv::xphoto::createGrayworldWB. Equivalent to
// [NewGrayworldWB].
func CreateGrayworldWB() *GrayworldWB { return NewGrayworldWB() }

// CreateLearningBasedWB returns a new [LearningBasedWB] with OpenCV's default
// parameters, mirroring cv::xphoto::createLearningBasedWB. Equivalent to
// [NewLearningBasedWB]. OpenCV's factory optionally loads a trained model file;
// this port has no trained model (see the package Deferred note), so the
// path-taking overload is intentionally omitted.
func CreateLearningBasedWB() *LearningBasedWB { return NewLearningBasedWB() }

// --- SimpleWB accessors ---------------------------------------------------

// SetInputMin sets the lower bound of the input range considered when building
// each channel histogram. Mirrors SimpleWB::setInputMin.
func (s *SimpleWB) SetInputMin(v float64) { s.InputMin = v }

// GetInputMin returns the lower bound of the input range. Mirrors
// SimpleWB::getInputMin.
func (s *SimpleWB) GetInputMin() float64 { return s.InputMin }

// SetInputMax sets the upper bound of the input range. Mirrors
// SimpleWB::setInputMax.
func (s *SimpleWB) SetInputMax(v float64) { s.InputMax = v }

// GetInputMax returns the upper bound of the input range. Mirrors
// SimpleWB::getInputMax.
func (s *SimpleWB) GetInputMax() float64 { return s.InputMax }

// SetOutputMin sets the lower bound of the output range each channel is
// stretched into. Mirrors SimpleWB::setOutputMin.
func (s *SimpleWB) SetOutputMin(v float64) { s.OutputMin = v }

// GetOutputMin returns the lower bound of the output range. Mirrors
// SimpleWB::getOutputMin.
func (s *SimpleWB) GetOutputMin() float64 { return s.OutputMin }

// SetOutputMax sets the upper bound of the output range. Mirrors
// SimpleWB::setOutputMax.
func (s *SimpleWB) SetOutputMax(v float64) { s.OutputMax = v }

// GetOutputMax returns the upper bound of the output range. Mirrors
// SimpleWB::getOutputMax.
func (s *SimpleWB) GetOutputMax() float64 { return s.OutputMax }

// SetP sets the tail-clip percentage (0..50) discarded from each channel before
// computing its stretch bounds. Mirrors SimpleWB::setP.
func (s *SimpleWB) SetP(v float64) { s.P = v }

// GetP returns the tail-clip percentage. Mirrors SimpleWB::getP.
func (s *SimpleWB) GetP() float64 { return s.P }

// --- GrayworldWB accessors ------------------------------------------------

// SetSaturationThreshold sets the (max-min)/max saturation above which a pixel
// is excluded from the gray-world statistics. Mirrors
// GrayworldWB::setSaturationThreshold.
func (g *GrayworldWB) SetSaturationThreshold(v float64) { g.SaturationThreshold = v }

// GetSaturationThreshold returns the saturation threshold. Mirrors
// GrayworldWB::getSaturationThreshold.
func (g *GrayworldWB) GetSaturationThreshold() float64 { return g.SaturationThreshold }

// --- LearningBasedWB accessors --------------------------------------------

// SetRangeMaxVal sets the maximum sample value (255 for 8-bit data). Mirrors
// LearningBasedWB::setRangeMaxVal.
func (l *LearningBasedWB) SetRangeMaxVal(v float64) { l.RangeMaxVal = v }

// GetRangeMaxVal returns the maximum sample value. Mirrors
// LearningBasedWB::getRangeMaxVal.
func (l *LearningBasedWB) GetRangeMaxVal() float64 { return l.RangeMaxVal }

// SetSaturationThreshold sets the fraction of RangeMaxVal above which a pixel's
// maximum channel is treated as clipped and excluded. Mirrors
// LearningBasedWB::setSaturationThreshold.
func (l *LearningBasedWB) SetSaturationThreshold(v float64) { l.SaturationThreshold = v }

// GetSaturationThreshold returns the saturation threshold fraction. Mirrors
// LearningBasedWB::getSaturationThreshold.
func (l *LearningBasedWB) GetSaturationThreshold() float64 { return l.SaturationThreshold }

// SetHistBinNum sets the internal chromaticity quantisation resolution. Mirrors
// LearningBasedWB::setHistBinNum.
func (l *LearningBasedWB) SetHistBinNum(v int) { l.HistBinNum = v }

// GetHistBinNum returns the chromaticity quantisation resolution. Mirrors
// LearningBasedWB::getHistBinNum.
func (l *LearningBasedWB) GetHistBinNum() int { return l.HistBinNum }
