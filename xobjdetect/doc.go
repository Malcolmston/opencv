// Package xobjdetect is a pure-Go port of OpenCV's xobjdetect module: a
// WaldBoost object detector built on integral-channel (ACF-style) features. It
// depends only on the Go standard library and the root
// [github.com/malcolmston/opencv] package (imported as cv); there is no cgo and
// there are no third-party dependencies.
//
// The module trains a boosted ensemble of decision stumps over integral
// channel features and evaluates it with a sequential-probability-ratio-test
// (SPRT) early-exit cascade, exactly the recipe behind OpenCV's
// cv::xobjdetect::WBDetector and its ICFDetector / ACFFeatureEvaluator
// machinery.
//
// # Integral channel features
//
// [ACFFeatureEvaluator] turns an image into ten feature channels — three
// colour channels (CIE L*a*b*, used as a stand-in for OpenCV's LUV), a
// gradient-magnitude channel, and six oriented-gradient histogram channels —
// and precomputes an integral image for each so that the mean value inside any
// axis-aligned rectangle of any channel is available in O(1). A single
// [Feature] names one such (channel, rectangle) probe; a [FeaturePool] is a
// deterministically sampled bank of them covering the detection window. This is
// the "integral channel features" representation of Dollár et al.
//
// # WaldBoost
//
// [WaldBoost] is the learning core. [WaldBoost.Train] fits a sequence of
// confidence-rated decision stumps with discrete AdaBoost, and after every
// stump records an SPRT rejection threshold on the running score. [WaldBoost.Predict]
// accumulates that score stump by stump and bails out the instant it drops
// below the current threshold, so obvious negatives are discarded after only a
// handful of features while true positives run the full ensemble.
//
// # WBDetector
//
// [WBDetector] ties the two together into a trainable, serialisable object
// detector. [WBDetector.Train] learns from positive and negative sample
// patches, [WBDetector.Detect] slides the learned classifier over a downscaling
// image pyramid and returns the accepted bounding boxes together with their
// confidence scores (after non-maximum suppression), and [WBDetector.Write] /
// [WBDetector.Read] round-trip a trained detector through [encoding/gob]
// without changing its behaviour.
//
// All geometry uses the root package's [cv.Rect] and [cv.Mat] types. Images may
// be single-channel (grayscale) or three-channel; the feature machinery
// normalises either to the same ten channels.
package xobjdetect
