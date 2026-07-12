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
//     stored as 336 bytes and compared by L1.
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
// OpenCV parameter. In particular the following are intentionally deferred:
//
//   - The full 72-bit, multi-orientation Marr–Hildreth descriptor with its
//     exact scale/orientation parameters; [MarrHildrethHash] implements a
//     simplified 64-bit LoG variant.
//   - OpenCV's peak-cross-correlation comparison for radial-variance hashes;
//     [RadialVarianceHash] uses the interface's L1 distance instead.
//   - Any learned or trained hashes (for example a calibrated descriptor
//     backed by a trained regressor).
package imghash
