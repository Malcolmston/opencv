// Package imghash is a from-scratch, standard-library-only port of a useful
// subset of OpenCV's img_hash contrib module: perceptual image hashing.
//
// A perceptual hash reduces an image to a short, fixed-length fingerprint that
// changes little under transformations a human would consider cosmetic —
// rescaling, mild blur, small brightness or contrast shifts, re-encoding — yet
// differs substantially between genuinely different images. Comparing two
// fingerprints is far cheaper than comparing two images and needs no pristine
// reference, which makes perceptual hashes the workhorse of near-duplicate
// detection, reverse image search and content-change monitoring.
//
// The package sits on top of the root module github.com/malcolmston/opencv
// (imported as cv) and the Go standard library (math, math/bits, sort,
// encoding/binary). It uses no cgo and no third-party dependencies, and it does
// not import any of the other cv/* subpackages. Every hasher operates on the
// package's central image type, [cv.Mat] (8-bit unsigned samples, one or three
// channels); colour images are reduced to grayscale with the BT.601 luma
// weights unless the hash is explicitly a colour descriptor.
//
// # The ImgHash interface
//
// Every hasher implements [ImgHash], mirroring OpenCV's
// cv::img_hash::ImgHashBase:
//
//	h := imghash.NewPHash()
//	a := h.Compute(imgA) // fixed-length []byte fingerprint
//	b := h.Compute(imgB)
//	d := h.Compare(a, b) // distance: 0 for identical, larger for different
//
// Compute returns a fresh byte slice whose length is fixed for the hasher.
// Compare returns the distance between two fingerprints from the same hasher —
// a Hamming (bit) distance for the binary hashes and an L1 distance for the
// real-valued ones. Every hasher type has a usable zero value and an idiomatic
// NewXxx constructor, plus a top-level convenience function that computes the
// hash in one call.
//
// # The hashes
//
//   - [AverageHash] (aHash), via [Average] — 8×8 mean threshold, 64 bits. The
//     simplest hash; robust to scale and uniform brightness.
//   - [PHash] (pHash), via [Perceptual] — 32×32 discrete cosine transform, the
//     top-left 8×8 low-frequency block thresholded at its median, 64 bits. The
//     most robust of the binary hashes to gamma, brightness and mild blur. The
//     2-D DCT is implemented locally (see [dct2D]).
//   - [DHash] (difference/gradient hash), via [Difference] — 9×8 horizontal
//     gradient signs, 64 bits. Naturally brightness- and contrast-invariant.
//   - [BlockMeanHash], via [BlockMean] — a grid of block means thresholded at
//     the global mean; 256 bits for the default 16×16 grid.
//   - [MarrHildrethHash], via [MarrHildreth] — a Laplacian-of-Gaussian filtered
//     image pooled into an 8×8 grid, 64 bits. Keys on edge structure.
//   - [RadialVarianceHash], via [RadialVariance] — the variance of Radon
//     projections at 40 orientations, normalised and quantised to 40 bytes,
//     compared by L1.
//   - [ColorMomentHash], via [ColorMoment] — the seven Hu moment invariants of
//     the R, G, B, H, S and V planes, a 42-dimensional real-valued descriptor
//     stored as 336 bytes and compared by L1. [ColorMomentL2Hash], via
//     [ColorMomentL2], computes the same 42-float descriptor but compares it by
//     the Euclidean (L2) distance, matching OpenCV's reported distance scale.
//
// # Additional hashes
//
//   - [MedianHash], via [Median] — the 8×8 aHash variant thresholded at the
//     block median rather than its mean, giving a perfectly balanced 64-bit
//     fingerprint.
//   - [AverageHashN], via [AverageN] — [AverageHash] generalised to an arbitrary
//     n×n grid, trading fingerprint length for spatial detail.
//   - [WaveletHash], via [Wavelet] — a wavelet hash keeping the 8×8 LL subband
//     of a three-level 2-D Haar transform of a 64×64 image, thresholded at its
//     median, 64 bits. The Haar transform is implemented locally and is exactly
//     invertible (see [HaarDWT2D] and [HaarIDWT2D]).
//   - [PHashN], via [PerceptualN] — [PHash] with a configurable k×k
//     low-frequency DCT block, so the fingerprint length and detail are tunable.
//   - [DHashVertical], via [DifferenceVertical] — the vertical-gradient
//     transpose of [DHash]; [DHashCombined], via [DifferenceCombined],
//     concatenates the horizontal and vertical difference hashes into a more
//     discriminative 128-bit fingerprint.
//   - [BlockMeanModeHash], via [BlockMeanMode] — the block mean hash with a
//     selectable projection mode ([BlockMeanMode0] non-overlapping,
//     [BlockMeanMode1] 50%-overlapping), matching OpenCV's mode enumeration.
//   - [MarrHildrethHash72], via [MarrHildreth72] — the full 72-bit, multi-scale,
//     multi-orientation Marr–Hildreth descriptor: the Laplacian-of-Gaussian
//     response at two Gaussian scales, steered into four orientation channels and
//     pooled into a 3×3 grid apiece (2×4×9 = 72 bits).
//   - [RadialVarianceCorrHash], via [RadialVarianceCorr] — the 40-byte
//     radial-variance fingerprint compared by OpenCV's peak cross-correlation
//     (see [PeakCrossCorrelation]) instead of L1, so it tolerates image rotation.
//
// # Encoding and thresholds
//
// [HexEncode] and [HexDecode] convert a fingerprint to and from its hexadecimal
// text form for storage or transport. [HammingNormalized] scales a binary
// hash's Hamming distance to [0, 1] so one threshold fits every length,
// [Similarity] reports 1 − that value as a "percentage alike" score, and
// [IsDuplicate] applies a normalised-distance threshold to flag near-duplicates.
//
// # Choosing a hash and a threshold
//
// For most near-duplicate work [PHash] or [DHash] is the best default: both are
// 64-bit and brightness-invariant, and a Hamming distance up to roughly 10 (out
// of 64) is a common "similar" cutoff, though the right threshold depends on the
// corpus and should be tuned. [AverageHash] is faster but weaker;
// [BlockMeanHash] trades a longer fingerprint for more spatial detail;
// [MarrHildrethHash] emphasises shape; and the real-valued [RadialVarianceHash]
// and [ColorMomentHash] capture orientation and colour distribution
// respectively.
//
// # Errors and panics
//
// Following the root package's conventions, the hashers panic on programmer
// error — a nil or empty image, or a Compare between fingerprints of different
// lengths (which means they came from different hashers) — rather than
// returning an error from every call.
//
// # Scope and deferred work
//
// This is a faithful port of the algorithms, not a bit-for-bit clone of every
// OpenCV parameter. The multi-scale, multi-orientation Marr–Hildreth descriptor
// is provided by [MarrHildrethHash72] (the simpler [MarrHildrethHash] remains as
// a 64-bit LoG variant), and OpenCV's peak-cross-correlation comparison for
// radial-variance hashes is provided by [RadialVarianceCorrHash] (the L1-based
// [RadialVarianceHash] is retained). The following remain intentionally deferred:
//
//   - Bit-exact reproduction of OpenCV's specific working sizes, kernel taps and
//     block layouts; the descriptors here follow the same algorithms with
//     independent, self-consistent parameters.
//   - Any learned or trained hashes (for example a calibrated descriptor
//     backed by a trained regressor).
package imghash
