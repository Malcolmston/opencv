# imghash2 — implements perceptual image hashing on top of the parent module's `cv.Mat` image type

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/imghash2.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/imghash2)

Package imghash2 implements perceptual image hashing on top of the parent
module's `cv.Mat` image type, using only the Go standard library.

A perceptual hash reduces an image to a short, fixed-length fingerprint whose
distance to another fingerprint tracks how visually similar the two images
are, unlike a cryptographic digest, which changes completely for any input
change. Two visually similar images (a rescaled, recompressed, slightly
blurred or brightness-shifted copy) produce hashes a small distance apart,
which makes these fingerprints the standard tool for near-duplicate detection,
reverse-image lookup and content deduplication.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/imghash2"
```

## Usage

This is the package's own `ExampleDCT2D`, so it compiles and its output is
asserted on every `go test ./imghash2/`.

```go
in := make([]float64, 16)
	for i := range in {
		in[i] = 5
	}
	c := imghash2.DCT2D(in, 4)
	fmt.Printf("DC=%.1f AC[1]=%.1f\n", c[0], c[1])
	back := imghash2.IDCT2D(c, 4)
	fmt.Printf("round trip=%.1f\n", back[7])
```

```
DC=20.0 AC[1]=0.0
round trip=5.0
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func DCT1D(vec []float64) []float64` | DCT1D returns the orthonormal type-II discrete cosine transform of vec: |
| `func DCT2D(in []float64, n int) []float64` | DCT2D returns the 2-D orthonormal type-II DCT of an n×n block stored row-major in a flat slice of length n·n. |
| `func FindDuplicates(hashes []Hash, maxDistance int) [][]int` | FindDuplicates groups a slice of equal-length hashes into clusters of near-duplicates. |
| `func GrayResize(img *cv.Mat, width, height int) *cv.Mat` | GrayResize converts img to grayscale with `ToGray` and rescales it to width×height using bilinear interpolation, the common front end of every… |
| `func HaarDWT1D(vec []float64) []float64` | HaarDWT1D applies one level of the orthonormal Haar discrete wavelet transform to vec and returns a fresh slice. |
| `func HaarDWT2D(in []float64, n int) []float64` | HaarDWT2D returns one level of the separable 2-D orthonormal Haar discrete wavelet transform of an n×n block stored row-major in a flat slice of… |
| `func HaarIDWT1D(vec []float64) []float64` | HaarIDWT1D is the exact inverse of `HaarDWT1D`, reconstructing the interleaved samples from the deinterleaved approximation and detail halves. |
| `func HaarIDWT2D(in []float64, n int) []float64` | HaarIDWT2D is the exact inverse of `HaarDWT2D`: given one level of 2-D Haar coefficients for an n×n block it reconstructs the original samples,… |
| `func HammingDistance(a, b Hash) int` | HammingDistance returns the number of differing bits between two equal-length binary hashes. |
| `func HexDecode(s string) ([]byte, error)` | HexDecode parses hexadecimal text into raw bytes, returning an error for invalid or odd-length input. |
| `func HexEncode(b []byte) string` | HexEncode returns the lower-case hexadecimal text form of raw hash bytes. |
| `func HuMoments(img *cv.Mat, c int) [7]float64` | HuMoments returns the seven Hu moment invariants of channel c of img, treating that channel's samples as an intensity image. |
| `func IDCT1D(coeffs []float64) []float64` | IDCT1D returns the inverse of `DCT1D` (the orthonormal type-III DCT), reconstructing the samples f(x) = Σ_u a(u) C(u) cos[(2x+1)uπ/(2N)]. |
| `func IDCT2D(in []float64, n int) []float64` | IDCT2D returns the inverse of `DCT2D`, reconstructing an n×n block from its coefficients. |
| `func IsDuplicate(a, b Hash, maxDistance int) bool` | IsDuplicate reports whether two equal-length binary hashes are near-duplicates under an absolute Hamming-distance threshold: it returns true when… |
| `func L1Distance(a, b FloatHash) float64` | L1Distance returns the L1 (Manhattan) distance between two equal-length descriptors, the free-function form of `FloatHash.L1`. |
| `func L2Distance(a, b FloatHash) float64` | L2Distance returns the L2 (Euclidean) distance between two equal-length descriptors, the free-function form of `FloatHash.L2`. |
| `func Mean(vals []float64) float64` | Mean returns the arithmetic mean of vals, or 0 for an empty slice. |
| `func Median(vals []float64) float64` | Median returns the median of vals. |
| `func NearDuplicate(a, b Hash, maxNormalizedDistance float64) bool` | NearDuplicate reports whether two equal-length binary hashes are near-duplicates under a normalised-distance threshold: it returns true when their… |
| `func NormalizedHamming(a, b Hash) float64` | NormalizedHamming returns the Hamming distance between two equal-length binary hashes divided by their bit count, a value in [0, 1]. |
| `func PopCount(b []byte) int` | PopCount returns the total number of set bits across all bytes of b, its Hamming weight. |
| `func RadialCrossCorrelation(a, b FloatHash) float64` | RadialCrossCorrelation returns the peak Pearson cross-correlation between two radial-variance descriptors over all circular shifts, a value in [-1,… |
| `func Similarity(a, b Hash) float64` | Similarity returns a normalised similarity score in [0, 1] between two binary hashes: 1 for identical fingerprints, falling toward 0 as bits diverge. |
| `func StdDev(vals []float64) float64` | StdDev returns the population standard deviation of vals, the square root of `Variance`. |
| `func ToGray(img *cv.Mat) *cv.Mat` | ToGray returns a single-channel version of img. |
| `func Variance(vals []float64) float64` | Variance returns the population variance of vals, or 0 for an empty slice. |

### Types

| Type | What it is |
| --- | --- |
| `AverageHash` | AverageHash implements the average hash (aHash) of Krawetz. |
| `BlockMeanHash` | BlockMeanHash implements a block mean-value hash in the spirit of Yang et al. |
| `ColorMomentHash` | ColorMomentHash implements a colour-moment hash in the spirit of OpenCV's cv::img_hash::ColorMomentHash. |
| `DifferenceHash` | DifferenceHash implements the difference (gradient) hash of Krawetz. |
| `Entry` | Entry associates an identifier with a hash, the input element of `DuplicatePairs`. |
| `FloatHash` | FloatHash is a real-valued perceptual descriptor: a fixed-length vector of float64 features returned by the descriptor hashes. |
| `FloatHasher` | FloatHasher is the interface implemented by the real-valued descriptor hashes (`RadialVarianceHash` and `ColorMomentHash`). |
| `Hash` | Hash is a binary perceptual fingerprint: a fixed-length bit string packed most-significant-bit first into bytes. |
| `Hasher` | Hasher is the common interface implemented by every binary perceptual hash in this package. |
| `Index` | Index is an incremental store of equal-length binary hashes that answers nearest-neighbour and radius queries. |
| `MarrHildrethHash` | MarrHildrethHash implements a Marr–Hildreth (Laplacian-of-Gaussian) hash in the spirit of Zauner's MH hash and OpenCV's… |
| `Match` | Match pairs a stored identifier with its Hamming distance to a query hash, the result element of `Index.Within` and `Index.Nearest`. |
| `PHash` | PHash implements the perceptual hash (pHash) of Zauner. |
| `Pair` | Pair names two identifiers found to be near-duplicates and the Hamming distance between their hashes, the result element of `DuplicatePairs`. |
| `RadialVarianceHash` | RadialVarianceHash implements a radial-variance hash in the spirit of the RASH descriptor and OpenCV's cv::img_hash::RadialVarianceHash. |
| `WaveletHash` | WaveletHash implements a wavelet hash (wHash) in the spirit of Krawetz's wavelet fingerprint. |

<details>
<summary><code>AverageHash</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewAverageHash() AverageHash` | NewAverageHash returns a ready-to-use `AverageHash`. |
| `func (AverageHash) Bits() int` | Bits returns 64, the fixed length of an average hash. |
| `func (AverageHash) Compute(img *cv.Mat) Hash` | Compute returns the 8-byte average hash of img. |
| `func (AverageHash) Name() string` | Name returns the identifier "ahash". |

</details>

<details>
<summary><code>BlockMeanHash</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewBlockMeanHash(blocks int) BlockMeanHash` | NewBlockMeanHash returns a `BlockMeanHash` using a blocks×blocks grid. |
| `func (h BlockMeanHash) Bits() int` | Bits returns the number of bits in the hash, the square of the grid resolution. |
| `func (h BlockMeanHash) Compute(img *cv.Mat) Hash` | Compute returns the block mean hash of img, Blocks·Blocks bits packed into ceil(Blocks·Blocks/8) bytes. |
| `func (BlockMeanHash) Name() string` | Name returns the identifier "blockmean". |

</details>

<details>
<summary><code>ColorMomentHash</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewColorMomentHash() ColorMomentHash` | NewColorMomentHash returns a ready-to-use `ColorMomentHash`. |
| `func (ColorMomentHash) ComputeFloat(img *cv.Mat) FloatHash` | ComputeFloat returns the 42-value colour-moment descriptor of img. |
| `func (ColorMomentHash) Name() string` | Name returns the identifier "colormoment". |

</details>

<details>
<summary><code>DifferenceHash</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewDifferenceHash() DifferenceHash` | NewDifferenceHash returns a horizontal `DifferenceHash`. |
| `func NewVerticalDifferenceHash() DifferenceHash` | NewVerticalDifferenceHash returns a vertical `DifferenceHash`. |
| `func (DifferenceHash) Bits() int` | Bits returns 64, the fixed length of a difference hash. |
| `func (h DifferenceHash) Compute(img *cv.Mat) Hash` | Compute returns the 8-byte difference hash of img. |
| `func (h DifferenceHash) Name() string` | Name returns "dhash" for the horizontal hash and "dhash-v" for the vertical hash. |

</details>

<details>
<summary><code>FloatHash</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func ColorMoment(img *cv.Mat) FloatHash` | ColorMoment is a convenience wrapper returning the `ColorMomentHash` descriptor of img. |
| `func ParseFloatHash(b []byte) FloatHash` | ParseFloatHash is the inverse of `FloatHash.Bytes`: it decodes big-endian float64 words back into a `FloatHash`. |
| `func RadialVariance(img *cv.Mat) FloatHash` | RadialVariance is a convenience wrapper returning the default `RadialVarianceHash` descriptor of img. |
| `func (f FloatHash) Bytes() []byte` | Bytes serialises the descriptor as consecutive big-endian IEEE-754 float64 words, a portable storage form recoverable with `ParseFloatHash`. |
| `func (f FloatHash) Clone() FloatHash` | Clone returns a fresh copy of the descriptor that shares no storage with the original. |
| `func (f FloatHash) Dims() int` | Dims returns the number of features in the descriptor. |
| `func (f FloatHash) L1(other FloatHash) float64` | L1 returns the L1 (Manhattan) distance between this descriptor and another of the same length: the sum of absolute per-feature differences. |
| `func (f FloatHash) L2(other FloatHash) float64` | L2 returns the L2 (Euclidean) distance between this descriptor and another of the same length. |

</details>

<details>
<summary><code>Hash</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func Average(img *cv.Mat) Hash` | Average is a convenience wrapper returning the `AverageHash` of img. |
| `func BlockMean(img *cv.Mat) Hash` | BlockMean is a convenience wrapper returning the default `BlockMeanHash` of img (a 16×16 grid, 32 bytes). |
| `func Difference(img *cv.Mat) Hash` | Difference is a convenience wrapper returning the horizontal `DifferenceHash` of img. |
| `func MarrHildreth(img *cv.Mat) Hash` | MarrHildreth is a convenience wrapper returning the `MarrHildrethHash` of img. |
| `func ParseHash(s string) (Hash, error)` | ParseHash decodes the hexadecimal text produced by `Hash.String` back into a `Hash`. |
| `func Perceptual(img *cv.Mat) Hash` | Perceptual is a convenience wrapper returning the default `PHash` of img. |
| `func Wavelet(img *cv.Mat) Hash` | Wavelet is a convenience wrapper returning the `WaveletHash` of img. |
| `func (h Hash) Bit(i int) bool` | Bit reports whether bit i of the hash is set, counting most-significant-bit first to match the packing used by the hashers. |
| `func (h Hash) Bits() int` | Bits returns the number of bits stored in the hash, which is eight times its byte length. |
| `func (h Hash) Bytes() []byte` | Bytes returns the underlying packed bytes of the hash. |
| `func (h Hash) Clone() Hash` | Clone returns a fresh copy of the hash that shares no storage with the original. |
| `func (h Hash) Equal(other Hash) bool` | Equal reports whether two hashes have the same length and identical bits. |
| `func (h Hash) Hamming(other Hash) int` | Hamming returns the Hamming distance — the number of differing bits — between this hash and another of the same length. |
| `func (h Hash) NormalizedHamming(other Hash) float64` | NormalizedHamming returns the Hamming distance divided by the bit count, a value in [0, 1] where 0 means identical and 1 means every bit differs. |
| `func (h Hash) OnesCount() int` | OnesCount returns the number of set bits in the hash (its Hamming weight). |
| `func (h Hash) Similarity(other Hash) float64` | Similarity returns 1 − `Hash.NormalizedHamming`, a score in [0, 1] that is 1 for identical fingerprints and falls toward 0 as bits diverge. |
| `func (h Hash) String() string` | String returns the lower-case hexadecimal text form of the hash, the usual way to store or transmit a fingerprint. |

</details>

<details>
<summary><code>Index</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewIndex() *Index` | NewIndex returns an empty `Index` ready to accept hashes. |
| `func (idx *Index) Add(id string, h Hash)` | Add inserts a hash under the given identifier. |
| `func (idx *Index) Bits() int` | Bits returns the bit length shared by every hash in the index, or 0 while the index is empty. |
| `func (idx *Index) Len() int` | Len returns the number of hashes stored in the index. |
| `func (idx *Index) Nearest(h Hash) (Match, bool)` | Nearest returns the stored entry closest to h in Hamming distance. |
| `func (idx *Index) Within(h Hash, maxDistance int) []Match` | Within returns every stored entry whose hash is at most maxDistance Hamming distance from h, in ascending order of distance (ties broken by insertion… |

</details>

<details>
<summary><code>MarrHildrethHash</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewMarrHildrethHash() MarrHildrethHash` | NewMarrHildrethHash returns a ready-to-use `MarrHildrethHash`. |
| `func (MarrHildrethHash) Bits() int` | Bits returns 64, the fixed length of a Marr–Hildreth hash. |
| `func (MarrHildrethHash) Compute(img *cv.Mat) Hash` | Compute returns the 8-byte Marr–Hildreth hash of img. |
| `func (MarrHildrethHash) Name() string` | Name returns the identifier "marrhildreth". |

</details>

<details>
<summary><code>PHash</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewPHash() PHash` | NewPHash returns a `PHash` using the default 32×32 working image. |
| `func NewPHashSize(size int) PHash` | NewPHashSize returns a `PHash` whose DCT runs on a size×size working image. |
| `func (PHash) Bits() int` | Bits returns 64, the fixed length of a perceptual hash. |
| `func (h PHash) Compute(img *cv.Mat) Hash` | Compute returns the 8-byte perceptual hash of img. |
| `func (h PHash) LowFrequencies(img *cv.Mat) []float64` | LowFrequencies returns the top-left 8×8 block of low-frequency DCT coefficients (row-major, 64 values) that `PHash.Compute` thresholds. |
| `func (PHash) Name() string` | Name returns the identifier "phash". |

</details>

<details>
<summary><code>Pair</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func DuplicatePairs(entries []Entry, maxDistance int) []Pair` | DuplicatePairs returns every pair of entries whose hashes are within maxDistance Hamming distance of each other. |

</details>

<details>
<summary><code>RadialVarianceHash</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewRadialVarianceHash() RadialVarianceHash` | NewRadialVarianceHash returns a ready-to-use `RadialVarianceHash` with the default 40 angles. |
| `func NewRadialVarianceHashAngles(angles int) RadialVarianceHash` | NewRadialVarianceHashAngles returns a `RadialVarianceHash` using the given number of projection angles. |
| `func (h RadialVarianceHash) ComputeFloat(img *cv.Mat) FloatHash` | ComputeFloat returns the radial-variance descriptor of img, a `FloatHash` of Angles values normalised so its peak is 1. |
| `func (RadialVarianceHash) Name() string` | Name returns the identifier "radialvariance". |

</details>

<details>
<summary><code>WaveletHash</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewWaveletHash() WaveletHash` | NewWaveletHash returns a ready-to-use `WaveletHash`. |
| `func (WaveletHash) Bits() int` | Bits returns 64, the fixed length of a wavelet hash. |
| `func (WaveletHash) Compute(img *cv.Mat) Hash` | Compute returns the 8-byte wavelet hash of img. |
| `func (WaveletHash) Name() string` | Name returns the identifier "whash". |

</details>

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/imghash2).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
