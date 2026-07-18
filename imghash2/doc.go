// Package imghash2 implements perceptual image hashing on top of the parent
// module's [cv.Mat] image type, using only the Go standard library.
//
// A perceptual hash reduces an image to a short, fixed-length fingerprint whose
// distance to another fingerprint tracks how visually similar the two images
// are, unlike a cryptographic digest, which changes completely for any input
// change. Two visually similar images (a rescaled, recompressed, slightly
// blurred or brightness-shifted copy) produce hashes a small distance apart,
// which makes these fingerprints the standard tool for near-duplicate
// detection, reverse-image lookup and content deduplication.
//
// # Hash types
//
// The package works with two fingerprint representations:
//
//   - [Hash] is a binary fingerprint (a bit string) compared by Hamming
//     distance. The bit hashes — [AverageHash] (aHash), [PHash] (pHash),
//     [DifferenceHash] (dHash), [WaveletHash] (wHash), [BlockMeanHash] and
//     [MarrHildrethHash] — all return one.
//   - [FloatHash] is a real-valued feature vector compared by L1 or L2
//     distance. The descriptor hashes — [RadialVarianceHash] and
//     [ColorMomentHash] — return one.
//
// The [Hasher] and [FloatHasher] interfaces unify the two families so callers
// can treat any hasher uniformly.
//
// # Choosing a hash
//
//   - [AverageHash] is the cheapest and most intuitive; robust to scale and
//     uniform brightness but weak against gamma and contrast changes.
//   - [PHash] works in the DCT frequency domain and thresholds at the median,
//     making it the most robust general-purpose choice.
//   - [DifferenceHash] encodes gradients, so it ignores absolute brightness and
//     complements the others cheaply.
//   - [WaveletHash] uses a spatially localised Haar basis, keying on where
//     structure lives as well as its frequency.
//   - [BlockMeanHash] captures coarse spatial layout at a tunable resolution.
//   - [MarrHildrethHash] emphasises edge structure via a Laplacian-of-Gaussian.
//   - [RadialVarianceHash] is a rotation-aware descriptor: comparing two of its
//     fingerprints with [RadialCrossCorrelation] is robust to image rotation.
//   - [ColorMomentHash] keys on the distribution of colour via Hu moments and
//     is invariant to translation, scale and rotation.
//
// # Near-duplicate detection
//
// [IsDuplicate] and [NearDuplicate] test a single pair against a threshold.
// [FindDuplicates] clusters a slice of hashes into groups of near-duplicates,
// and [Index] provides an incremental store with nearest-neighbour and
// radius queries backed by a BK-tree for sub-linear lookups on binary hashes.
//
// # Building blocks
//
// The signal transforms the hashes are built on are exported for reuse and
// independent testing: [DCT1D]/[DCT2D] and their inverses, and the orthonormal
// Haar wavelet transform [HaarDWT2D]/[HaarIDWT2D]. Small numeric helpers
// ([Mean], [Median], [Variance], [StdDev]) and the [HuMoments] shape invariants
// are exported as well.
//
// # Conventions
//
// Every hasher first reduces its input to grayscale (BT.601 luma for
// three-channel RGB, matching the parent package) and rescales it, so the
// fingerprints are invariant to the source resolution. All routines are
// deterministic, allocation-friendly and free of any third-party dependency,
// cgo or GPU code.
package imghash2
