# imghash — from-scratch, standard-library-only port of a useful subset of OpenCV's img_hash

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/imghash.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/imghash)

Package imghash is a from-scratch, standard-library-only port of a useful
subset of OpenCV's img_hash contrib module: perceptual image hashing.

A perceptual hash reduces an image to a short, fixed-length fingerprint that
changes little under transformations a human would consider cosmetic —
rescaling, mild blur, small brightness or contrast shifts, re-encoding — yet
differs substantially between genuinely different images. Comparing two
fingerprints is far cheaper than comparing two images and needs no pristine
reference, which makes perceptual hashes the workhorse of near-duplicate
detection, reverse image search and content-change monitoring.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/imghash"
```

## Usage

This is the package's own `ExampleAverageHash`, so it compiles and its output is
asserted on every `go test ./imghash/`.

```go
img := ramp(32, 32)
	h := imghash.NewAverageHash()
	a := h.Compute(img)
	b := h.Compute(img.Clone())
	fmt.Printf("bytes=%d distance=%.0f\n", len(a), h.Compare(a, b))
```

```
bytes=8 distance=0
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func Average(img *cv.Mat) []byte` | Average is a convenience wrapper returning the `AverageHash` of img. |
| `func AverageN(img *cv.Mat, n int) []byte` | AverageN is a convenience wrapper returning the n×n `AverageHashN` of img. |
| `func BlockMean(img *cv.Mat) []byte` | BlockMean is a convenience wrapper returning the default `BlockMeanHash` of img (a 16×16 grid, 32 bytes). |
| `func BlockMeanMode(img *cv.Mat, mode int) []byte` | BlockMeanMode is a convenience wrapper returning the default 16×16 `BlockMeanModeHash` of img in the given mode. |
| `func ColorMoment(img *cv.Mat) []byte` | ColorMoment is a convenience wrapper returning the `ColorMomentHash` of img. |
| `func ColorMomentL2(img *cv.Mat) []byte` | ColorMomentL2 is a convenience wrapper returning the `ColorMomentL2Hash` of img. |
| `func Difference(img *cv.Mat) []byte` | Difference is a convenience wrapper returning the `DHash` of img. |
| `func DifferenceCombined(img *cv.Mat) []byte` | DifferenceCombined is a convenience wrapper returning the `DHashCombined` of img. |
| `func DifferenceVertical(img *cv.Mat) []byte` | DifferenceVertical is a convenience wrapper returning the `DHashVertical` of img. |
| `func HaarDWT2D(in []float64, n int) []float64` | HaarDWT2D returns one level of the separable 2-D orthonormal Haar discrete wavelet transform of an n×n block stored row-major in a flat slice of… |
| `func HaarIDWT2D(in []float64, n int) []float64` | HaarIDWT2D is the exact inverse of `HaarDWT2D`: given one level of 2-D Haar coefficients for an n×n block it reconstructs the original samples. |
| `func HammingNormalized(a, b []byte) float64` | HammingNormalized returns the Hamming distance between two equal-length binary hashes divided by their bit count, a value in [0, 1] where 0 means… |
| `func HexDecode(s string) ([]byte, error)` | HexDecode parses the hexadecimal text produced by `HexEncode` back into hash bytes. |
| `func HexEncode(h []byte) string` | HexEncode returns the lower-case hexadecimal text form of a hash, the usual way to store or transmit a fingerprint. |
| `func IsDuplicate(a, b []byte, maxNormalizedDistance float64) bool` | IsDuplicate reports whether two binary hashes are near-duplicates under a normalised-distance threshold: it returns true when `HammingNormalized` of… |
| `func MarrHildreth(img *cv.Mat) []byte` | MarrHildreth is a convenience wrapper returning the `MarrHildrethHash` of img. |
| `func MarrHildreth72(img *cv.Mat) []byte` | MarrHildreth72 is a convenience wrapper returning the `MarrHildrethHash72` of img. |
| `func Median(img *cv.Mat) []byte` | Median is a convenience wrapper returning the `MedianHash` of img. |
| `func PeakCrossCorrelation(a, b []float64) float64` | PeakCrossCorrelation returns the peak normalised cross-correlation between two equal-length real vectors over all circular shifts, a value in [-1, 1]. |
| `func Perceptual(img *cv.Mat) []byte` | Perceptual is a convenience wrapper returning the `PHash` of img. |
| `func PerceptualN(img *cv.Mat, k int) []byte` | PerceptualN is a convenience wrapper returning the `PHashN` of img keeping the top-left k×k DCT block. |
| `func RadialVariance(img *cv.Mat) []byte` | RadialVariance is a convenience wrapper returning the `RadialVarianceHash` of img. |
| `func RadialVarianceCorr(img *cv.Mat) []byte` | RadialVarianceCorr is a convenience wrapper returning the `RadialVarianceCorrHash` of img. |
| `func Similarity(a, b []byte) float64` | Similarity returns a normalised similarity score in [0, 1] between two binary hashes: 1 for identical fingerprints, falling to 0 as bits diverge. |
| `func Wavelet(img *cv.Mat) []byte` | Wavelet is a convenience wrapper returning the `WaveletHash` of img. |

### Types

| Type | What it is |
| --- | --- |
| `AverageHash` | AverageHash implements the average hash (aHash) of Krawetz. |
| `AverageHashN` | AverageHashN generalises `AverageHash` to an arbitrary n×n grid. |
| `BlockMeanHash` | BlockMeanHash implements a block mean-value hash in the spirit of Yang et al. |
| `BlockMeanModeHash` | BlockMeanModeHash implements the block mean-value hash with a selectable projection mode, matching OpenCV's cv::img_hash::BlockMeanHash more closely… |
| `ColorMomentHash` | ColorMomentHash implements a colour-moment hash in the spirit of OpenCV's cv::img_hash::ColorMomentHash. |
| `ColorMomentL2Hash` | ColorMomentL2Hash computes the same full 42-float colour-moment descriptor as `ColorMomentHash` — the seven log-scaled Hu moment invariants of the… |
| `DHash` | DHash implements the difference (gradient) hash of Krawetz. |
| `DHashCombined` | DHashCombined concatenates the horizontal difference hash of `DHash` with the vertical difference hash of `DHashVertical`, producing a 128-bit… |
| `DHashVertical` | DHashVertical implements the vertical difference (gradient) hash, the transpose of `DHash`. |
| `ImgHash` | ImgHash is the common interface implemented by every perceptual hash in this package, mirroring OpenCV's cv::img_hash::ImgHashBase. |
| `MarrHildrethHash` | MarrHildrethHash implements a Marr–Hildreth (Laplacian-of-Gaussian) hash in the spirit of Zauner's MH hash and OpenCV's… |
| `MarrHildrethHash72` | MarrHildrethHash72 implements the full 72-bit, multi-scale, multi-orientation Marr–Hildreth descriptor deferred by the simplified 64-bit… |
| `MedianHash` | MedianHash implements the median hash, a close relative of `AverageHash`. |
| `PHash` | PHash implements the perceptual hash (pHash) of Zauner. |
| `PHashN` | PHashN generalises `PHash` to a configurable low-frequency block size. |
| `RadialVarianceCorrHash` | RadialVarianceCorrHash computes the same 40-byte radial-variance fingerprint as `RadialVarianceHash` but compares two fingerprints by OpenCV's peak… |
| `RadialVarianceHash` | RadialVarianceHash implements a radial-variance hash in the spirit of the RASH descriptor and OpenCV's cv::img_hash::RadialVarianceHash. |
| `WaveletHash` | WaveletHash implements a wavelet hash (wHash) in the spirit of Krawetz's wavelet fingerprint. |

<details>
<summary><code>AverageHash</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewAverageHash() AverageHash` | NewAverageHash returns a ready-to-use `AverageHash`. |
| `func (AverageHash) Compare(a, b []byte) float64` | Compare returns the Hamming distance between two average hashes. |
| `func (AverageHash) Compute(img *cv.Mat) []byte` | Compute returns the 8-byte average hash of img. |

</details>

<details>
<summary><code>AverageHashN</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewAverageHashN(n int) AverageHashN` | NewAverageHashN returns an `AverageHashN` using an n×n grid. |
| `func (AverageHashN) Compare(a, b []byte) float64` | Compare returns the Hamming distance between two average hashes. |
| `func (h AverageHashN) Compute(img *cv.Mat) []byte` | Compute returns the n²-bit average hash of img. |

</details>

<details>
<summary><code>BlockMeanHash</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewBlockMeanHash(blocks int) BlockMeanHash` | NewBlockMeanHash returns a `BlockMeanHash` using a blocks×blocks grid. |
| `func (BlockMeanHash) Compare(a, b []byte) float64` | Compare returns the Hamming distance between two block mean hashes. |
| `func (h BlockMeanHash) Compute(img *cv.Mat) []byte` | Compute returns the block mean hash of img. |

</details>

<details>
<summary><code>BlockMeanModeHash</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewBlockMeanModeHash(blocks, mode int) BlockMeanModeHash` | NewBlockMeanModeHash returns a `BlockMeanModeHash` using a blocks×blocks grid and the given projection mode (`BlockMeanMode0` or `BlockMeanMode1`). |
| `func (BlockMeanModeHash) Compare(a, b []byte) float64` | Compare returns the Hamming distance between two block mean hashes. |
| `func (h BlockMeanModeHash) Compute(img *cv.Mat) []byte` | Compute returns the block mean hash of img for the configured grid and mode. |

</details>

<details>
<summary><code>ColorMomentHash</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewColorMomentHash() ColorMomentHash` | NewColorMomentHash returns a ready-to-use `ColorMomentHash`. |
| `func (ColorMomentHash) Compare(a, b []byte) float64` | Compare returns the L1 distance between two colour-moment hashes. |
| `func (ColorMomentHash) Compute(img *cv.Mat) []byte` | Compute returns the 336-byte colour-moment hash of img (42 float64 values). |

</details>

<details>
<summary><code>ColorMomentL2Hash</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewColorMomentL2Hash() ColorMomentL2Hash` | NewColorMomentL2Hash returns a ready-to-use `ColorMomentL2Hash`. |
| `func (ColorMomentL2Hash) Compare(a, b []byte) float64` | Compare returns the Euclidean (L2) distance between two colour-moment hashes. |
| `func (ColorMomentL2Hash) Compute(img *cv.Mat) []byte` | Compute returns the 336-byte colour-moment hash of img (42 float64 values), identical to `ColorMomentHash.Compute`. |

</details>

<details>
<summary><code>DHash</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewDHash() DHash` | NewDHash returns a ready-to-use `DHash`. |
| `func (DHash) Compare(a, b []byte) float64` | Compare returns the Hamming distance between two difference hashes. |
| `func (DHash) Compute(img *cv.Mat) []byte` | Compute returns the 8-byte difference hash of img. |

</details>

<details>
<summary><code>DHashCombined</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewDHashCombined() DHashCombined` | NewDHashCombined returns a ready-to-use `DHashCombined`. |
| `func (DHashCombined) Compare(a, b []byte) float64` | Compare returns the Hamming distance between two combined difference hashes. |
| `func (DHashCombined) Compute(img *cv.Mat) []byte` | Compute returns the 16-byte combined difference hash of img: the eight bytes of the horizontal hash followed by the eight bytes of the vertical hash. |

</details>

<details>
<summary><code>DHashVertical</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewDHashVertical() DHashVertical` | NewDHashVertical returns a ready-to-use `DHashVertical`. |
| `func (DHashVertical) Compare(a, b []byte) float64` | Compare returns the Hamming distance between two vertical difference hashes. |
| `func (DHashVertical) Compute(img *cv.Mat) []byte` | Compute returns the 8-byte vertical difference hash of img. |

</details>

<details>
<summary><code>MarrHildrethHash</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewMarrHildrethHash() MarrHildrethHash` | NewMarrHildrethHash returns a ready-to-use `MarrHildrethHash`. |
| `func (MarrHildrethHash) Compare(a, b []byte) float64` | Compare returns the Hamming distance between two Marr–Hildreth hashes. |
| `func (MarrHildrethHash) Compute(img *cv.Mat) []byte` | Compute returns the 8-byte Marr–Hildreth hash of img. |

</details>

<details>
<summary><code>MarrHildrethHash72</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewMarrHildrethHash72() MarrHildrethHash72` | NewMarrHildrethHash72 returns a ready-to-use `MarrHildrethHash72`. |
| `func (MarrHildrethHash72) Compare(a, b []byte) float64` | Compare returns the Hamming distance between two 72-bit Marr–Hildreth hashes. |
| `func (MarrHildrethHash72) Compute(img *cv.Mat) []byte` | Compute returns the 9-byte (72-bit) multi-scale, multi-orientation Marr–Hildreth hash of img. |

</details>

<details>
<summary><code>MedianHash</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewMedianHash() MedianHash` | NewMedianHash returns a ready-to-use `MedianHash`. |
| `func (MedianHash) Compare(a, b []byte) float64` | Compare returns the Hamming distance between two median hashes. |
| `func (MedianHash) Compute(img *cv.Mat) []byte` | Compute returns the 8-byte median hash of img. |

</details>

<details>
<summary><code>PHash</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewPHash() PHash` | NewPHash returns a ready-to-use `PHash`. |
| `func (PHash) Compare(a, b []byte) float64` | Compare returns the Hamming distance between two perceptual hashes. |
| `func (PHash) Compute(img *cv.Mat) []byte` | Compute returns the 8-byte perceptual hash of img. |

</details>

<details>
<summary><code>PHashN</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewPHashN(k int) PHashN` | NewPHashN returns a `PHashN` keeping the top-left k×k DCT block. |
| `func (PHashN) Compare(a, b []byte) float64` | Compare returns the Hamming distance between two perceptual hashes. |
| `func (h PHashN) Compute(img *cv.Mat) []byte` | Compute returns the k²-bit perceptual hash of img. |

</details>

<details>
<summary><code>RadialVarianceCorrHash</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewRadialVarianceCorrHash() RadialVarianceCorrHash` | NewRadialVarianceCorrHash returns a ready-to-use `RadialVarianceCorrHash`. |
| `func (RadialVarianceCorrHash) Compare(a, b []byte) float64` | Compare returns 1 − the peak cross-correlation of the two hashes, a distance in roughly [0, 2] that is 0 for identical (or purely rotated) images. |
| `func (RadialVarianceCorrHash) Compute(img *cv.Mat) []byte` | Compute returns the 40-byte radial-variance hash of img, identical to `RadialVarianceHash.Compute`. |

</details>

<details>
<summary><code>RadialVarianceHash</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewRadialVarianceHash() RadialVarianceHash` | NewRadialVarianceHash returns a ready-to-use `RadialVarianceHash`. |
| `func (RadialVarianceHash) Compare(a, b []byte) float64` | Compare returns the L1 distance between two radial-variance hashes. |
| `func (RadialVarianceHash) Compute(img *cv.Mat) []byte` | Compute returns the 40-byte radial-variance hash of img. |

</details>

<details>
<summary><code>WaveletHash</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewWaveletHash() WaveletHash` | NewWaveletHash returns a ready-to-use `WaveletHash`. |
| `func (WaveletHash) Compare(a, b []byte) float64` | Compare returns the Hamming distance between two wavelet hashes. |
| `func (WaveletHash) Compute(img *cv.Mat) []byte` | Compute returns the 8-byte wavelet hash of img. |

</details>

### Constants

`BlockMeanMode0`, `BlockMeanMode1`

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/imghash).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
