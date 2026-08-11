# texture — implements classic image-texture analysis on top of the root package's `cv.Mat` image

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/texture.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/texture)

Package texture implements classic image-texture analysis on top of the root
package's `cv.Mat` image type, using only the Go standard library.

Texture is the spatial arrangement of intensities in an image — the visual
"feel" of a surface such as grass, brick, fabric or a medical scan. Unlike
colour or shape, texture is a statistical property of a neighbourhood rather
than of a single pixel, so every routine here summarises how gray levels are
distributed and co-arranged across a region.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/texture"
```

## Usage

This is the package's own `ExampleBoxCountingDimension`, so it compiles and its output is
asserted on every `go test ./texture/`.

```go
solid := cv.NewMat(32, 32, 1)
	for i := range solid.Data {
		solid.Data[i] = 255
	}
	sparse := cv.NewMat(32, 32, 1)
	for y := 0; y < 32; y++ {
		sparse.Data[y*32+y] = 255
	}

	d1 := texture.BoxCountingDimension(solid, 128)
	d2 := texture.BoxCountingDimension(sparse, 128)
	fmt.Printf("solid block fills more space than a line: %v\n", d1 > d2)
```

```
solid block fills more space than a line: true
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func BoxCountingDimension(img *cv.Mat, threshold uint8) float64` | BoxCountingDimension estimates the Minkowski–Bouligand (box-counting) fractal dimension of the foreground of img. |
| `func DifferentialBoxCounting(img *cv.Mat) float64` | DifferentialBoxCounting estimates the fractal dimension of a grayscale image with the Sarkar–Chaudhuri differential box-counting method, which… |
| `func FractalDimension(img *cv.Mat) float64` | FractalDimension is a convenience alias for `DifferentialBoxCounting`: it returns a grayscale fractal-dimension estimate of img requiring no… |
| `func GaborEnergy(img *cv.Mat, size int, p GaborParams) float64` | GaborEnergy returns the mean absolute Gabor response of img for the filter described by p with a size-by-size kernel — a single scalar summarising… |
| `func GaborFeatures(img *cv.Mat, size int, p GaborParams, nOrient int, wavelengths []float64) []float64` | GaborFeatures runs a bank of nOrient orientations at the given wavelengths over img and returns a feature vector holding, for each filter, the mean… |
| `func GaborFilter(img *cv.Mat, size int, p GaborParams) *cv.Mat` | GaborFilter convolves img with the Gabor kernel described by p and returns the filtered response as a single-channel `cv.Mat`. |
| `func GaborFilterBank(size int, p GaborParams, nOrient int, wavelengths []float64) [][]float64` | GaborFilterBank builds a bank of Gabor kernels sharing base parameters p but spanning nOrient equally spaced orientations in [0, pi) at each of the… |
| `func GaborKernel(size int, p GaborParams) []float64` | GaborKernel builds a size-by-size real Gabor kernel from the given parameters and returns it in row-major order. |
| `func GaborMagnitude(img *cv.Mat, size int, p GaborParams) [][]float64` | GaborMagnitude returns the local Gabor magnitude field of img: the pixelwise square root of the sum of squares of the responses to the even (cosine,… |
| `func Histogram(labels [][]int, bins int) []float64` | Histogram tallies the labels of a grid (as returned by `LBPUniform` or `LBPUniformRotationInvariant`) into a normalised histogram with bins bins. |
| `func IsUniform(code uint32, points int) bool` | IsUniform reports whether the low `points` bits of code form a uniform pattern, i.e. |
| `func LBP(img *cv.Mat, radius, points int) [][]uint32` | LBP computes the Local Binary Pattern code for every pixel of img and returns the codes as a rows-by-cols grid (row-major, one uint32 per pixel). |
| `func LBPImage(img *cv.Mat) *cv.Mat` | LBPImage computes the classic 3x3, 8-neighbour Local Binary Pattern and returns it as a single-channel `cv.Mat` whose samples are the 8-bit codes. |
| `func LBPRotationInvariant(img *cv.Mat, radius, points int) [][]uint32` | LBPRotationInvariant computes rotation-invariant LBP codes for every pixel of img via `RotateMinimum` and returns them as a rows-by-cols grid. |
| `func LBPRotationInvariantUniformHistogram(img *cv.Mat, radius, points int) []float64` | LBPRotationInvariantUniformHistogram returns the normalised rotation-invariant uniform ("riu2") histogram of img, with points+2 entries. |
| `func LBPUniform(img *cv.Mat, radius, points int) [][]int` | LBPUniform computes uniform LBP labels for every pixel of img and returns them as a rows-by-cols grid. |
| `func LBPUniformHistogram(img *cv.Mat, radius, points int) []float64` | LBPUniformHistogram is a convenience routine returning the normalised uniform LBP histogram of img: it computes uniform labels with `LBPUniform` and… |
| `func LBPUniformRotationInvariant(img *cv.Mat, radius, points int) [][]int` | LBPUniformRotationInvariant computes the rotation-invariant uniform ("riu2") label for every pixel and returns them as a rows-by-cols grid, each… |
| `func Lacunarity(img *cv.Mat, threshold uint8, boxSize int) float64` | Lacunarity computes the gliding-box lacunarity of the foreground of img at the given box size, a measure of how "gappy" or heterogeneous a fractal… |
| `func LawsE5() []float64` | LawsE5 returns the length-5 Edge mask [-1 -2 0 2 1], a first-derivative (edge) filter. |
| `func LawsFeatures(img *cv.Mat, window int) []float64` | LawsFeatures returns a compact rotation-invariant Laws texture feature vector: the image-wide mean of each of the nine standard combined energy maps… |
| `func LawsKernel2D(v, h []float64) []float64` | LawsKernel2D forms the separable 2-D Laws kernel as the outer product of a vertical mask v and a horizontal mask h (each typically length 5),… |
| `func LawsL5() []float64` | LawsL5 returns the length-5 Level mask [1 4 6 4 1], a smoothing (local average) filter. |
| `func LawsR5() []float64` | LawsR5 returns the length-5 Ripple mask [1 -4 6 -4 1]. |
| `func LawsS5() []float64` | LawsS5 returns the length-5 Spot mask [-1 0 2 0 -1], a second-derivative (spot) filter. |
| `func LawsW5() []float64` | LawsW5 returns the length-5 Wave mask [-1 2 0 -2 1]. |
| `func MapUniform(code uint32, points int) int` | MapUniform maps an LBP code to its uniform-LBP bin index in [0, UniformBinCount(points)). |
| `func MapUniformRotationInvariant(code uint32, points int) int` | MapUniformRotationInvariant maps an LBP code to its rotation-invariant uniform label in [0, points+1]: a uniform pattern is labelled by its number of… |
| `func RotateMinimum(code uint32, points int) uint32` | RotateMinimum returns the rotation-invariant canonical form of the low `points` bits of code: the numerically smallest value obtainable by any… |
| `func TamuraCoarseness(img *cv.Mat, kmax int) float64` | TamuraCoarseness returns the Tamura coarseness of img. |
| `func TamuraContrast(img *cv.Mat) float64` | TamuraContrast returns the Tamura contrast of img, defined as the standard deviation divided by the fourth root of the kurtosis (sigma / alpha4^0.25,… |
| `func TamuraDirectionality(img *cv.Mat, nbins int, threshold float64) float64` | TamuraDirectionality returns the Tamura directionality of img in [0,1]. |
| `func UniformBinCount(points int) int` | UniformBinCount returns the number of histogram bins used by the uniform LBP mapping for the given neighbour count: one bin per uniform pattern… |

### Types

| Type | What it is |
| --- | --- |
| `Direction` | Direction identifies one of the four canonical neighbour orientations used to build a co-occurrence or run-length matrix. |
| `GLCM` | GLCM is a gray-level co-occurrence matrix: a levels-by-levels table whose entry (i, j) counts how often a pixel of quantised gray level i is found… |
| `GaborParams` | GaborParams holds the parameters of a 2-D Gabor filter — a sinusoidal plane wave modulated by a Gaussian envelope, the classic model of oriented,… |
| `HaralickFeatures` | HaralickFeatures bundles the standard scalar texture features derived from a single `GLCM`. |
| `LawsEnergy` | LawsEnergy holds the texture-energy maps produced by `LawsEnergyMaps`. |
| `RunLengthFeatures` | RunLengthFeatures bundles the eleven standard Galloway run-length statistics. |
| `RunLengthMatrix` | RunLengthMatrix is a gray-level run-length matrix (GLRLM): entry (i, j) counts the number of runs — maximal sequences of collinear, consecutive… |
| `TamuraFeatureSet` | TamuraFeatureSet bundles the three principal Tamura perceptual texture features together with the derived roughness. |

<details>
<summary><code>Direction</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (dir Direction) Offset(distance int) (dx, dy int)` | Offset returns the pixel column/row displacement (dx, dy) for this direction at the given distance. |
| `func (dir Direction) String() string` | String returns a human-readable name for the direction. |

</details>

<details>
<summary><code>GLCM</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func ComputeGLCM(img *cv.Mat, levels int, dir Direction, distance int, symmetric bool) *GLCM` | ComputeGLCM is a convenience wrapper over `NewGLCM` that takes a symbolic `Direction` and a distance instead of a raw offset. |
| `func NewGLCM(img *cv.Mat, levels, dx, dy int, symmetric bool) *GLCM` | NewGLCM builds a co-occurrence matrix from img using an explicit pixel offset (dx, dy). |
| `func (g *GLCM) ASM() float64` | ASM returns the angular second moment, the sum of p(i,j)^2. |
| `func (g *GLCM) At(i, j int) float64` | At returns the raw co-occurrence count for the ordered gray-level pair (i, j). |
| `func (g *GLCM) Autocorrelation() float64` | Autocorrelation returns the sum of i*j*p(i,j), a simple measure of the fineness and coarseness of texture. |
| `func (g *GLCM) ClusterProminence() float64` | ClusterProminence returns the fourth-order cluster prominence, sum((i+j-mux-muy)^4 p(i,j)); large values indicate an asymmetric, heavy-tailed… |
| `func (g *GLCM) ClusterShade() float64` | ClusterShade returns the third-order cluster shade, sum((i+j-mux-muy)^3 p(i,j)). |
| `func (g *GLCM) Contrast() float64` | Contrast returns the Haralick contrast (a.k.a. |
| `func (g *GLCM) Correlation() float64` | Correlation returns the Haralick correlation, a measure in [-1, 1] of the linear dependency of gray levels between neighbouring pixels. |
| `func (g *GLCM) DifferenceAverage() float64` | DifferenceAverage returns sum(k * p_{x-y}(k)), the mean absolute gray-level difference between neighbouring pixels. |
| `func (g *GLCM) DifferenceEntropy() float64` | DifferenceEntropy returns the entropy -sum p_{x-y}(k) ln p_{x-y}(k) of the gray-level-difference distribution (natural log). |
| `func (g *GLCM) DifferenceVariance() float64` | DifferenceVariance returns the variance of the gray-level-difference distribution p_{x-y}, centred on its own mean `GLCM.DifferenceAverage`. |
| `func (g *GLCM) Dissimilarity() float64` | Dissimilarity returns the sum of \|i-j\| * p(i,j), a linear analogue of contrast that weights large gray-level differences less heavily. |
| `func (g *GLCM) Energy() float64` | Energy returns sqrt(`GLCM.ASM`), the L2 norm of the probability matrix. |
| `func (g *GLCM) Entropy() float64` | Entropy returns the Shannon entropy -sum p(i,j)*ln p(i,j) (natural log, nats). |
| `func (g *GLCM) Haralick() HaralickFeatures` | Haralick computes the full bundle of scalar features in a single pass-worth of calls and returns them as a `HaralickFeatures` value. |
| `func (g *GLCM) Homogeneity() float64` | Homogeneity returns the inverse difference moment, the sum of p(i,j)/(1+(i-j)^2). |
| `func (g *GLCM) Levels() int` | Levels returns the number of gray levels (the matrix side length). |
| `func (g *GLCM) MaximumProbability() float64` | MaximumProbability returns the largest normalised entry of the matrix, the probability of the most common gray-level pair. |
| `func (g *GLCM) Probabilities() []float64` | Probabilities returns a fresh copy of the matrix normalised to sum to 1, in row-major order. |
| `func (g *GLCM) Sum() float64` | Sum returns the total number of gray-level pairs tallied in the matrix. |
| `func (g *GLCM) SumAverage() float64` | SumAverage returns sum(k * p_{x+y}(k)), the mean of the distribution of neighbouring gray-level sums. |
| `func (g *GLCM) SumEntropy() float64` | SumEntropy returns the entropy -sum p_{x+y}(k) ln p_{x+y}(k) of the gray-level-sum distribution (natural log). |
| `func (g *GLCM) SumVariance() float64` | SumVariance returns the Haralick sum variance, sum((k - SumEntropy)^2 p_{x+y}(k)), using the sum entropy as the centre per Haralick's original… |
| `func (g *GLCM) Variance() float64` | Variance returns the Haralick variance sum((i-mu)^2 p(i,j)) where mu is the overall matrix mean of the row index; it quantifies the spread of gray… |

</details>

<details>
<summary><code>LawsEnergy</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func LawsEnergyMaps(img *cv.Mat, window int) *LawsEnergy` | LawsEnergyMaps computes Laws texture-energy maps for img. |
| `func (le *LawsEnergy) Combined(a, b string) []float64` | Combined returns the rotation-invariant energy map for an unordered mask pair by averaging the two ordered maps (for example "E5L5" and "L5E5"). |
| `func (le *LawsEnergy) Map(name string) []float64` | Map returns the energy map for the ordered mask pair name (for example "S5S5"), or nil if that pair was not computed. |

</details>

<details>
<summary><code>RunLengthMatrix</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewRunLengthMatrix(img *cv.Mat, levels int, dir Direction) *RunLengthMatrix` | NewRunLengthMatrix builds the run-length matrix of img along dir. |
| `func (rl *RunLengthMatrix) At(i, j int) float64` | At returns the number of runs of gray level i (0-based) and length j (1-based). |
| `func (rl *RunLengthMatrix) Features() RunLengthFeatures` | Features computes all eleven run-length statistics and returns them as a `RunLengthFeatures` value. |
| `func (rl *RunLengthMatrix) GrayLevelNonUniformity() float64` | GrayLevelNonUniformity returns GLN = (1/Nr) sum_i (sum_j P(i,j))^2, which is low when runs are spread evenly across gray levels and high when a few… |
| `func (rl *RunLengthMatrix) HighGrayLevelRunEmphasis() float64` | HighGrayLevelRunEmphasis returns HGRE = (1/Nr) sum P(i,j)*(i+1)^2, emphasising runs of high gray level. |
| `func (rl *RunLengthMatrix) Levels() int` | Levels returns the number of gray levels. |
| `func (rl *RunLengthMatrix) LongRunEmphasis() float64` | LongRunEmphasis returns LRE = (1/Nr) sum P(i,j)*j^2, large for textures dominated by long runs (coarse texture). |
| `func (rl *RunLengthMatrix) LongRunHighGrayLevelEmphasis() float64` | LongRunHighGrayLevelEmphasis returns LRHGE = (1/Nr) sum P(i,j)*j^2*(i+1)^2. |
| `func (rl *RunLengthMatrix) LongRunLowGrayLevelEmphasis() float64` | LongRunLowGrayLevelEmphasis returns LRLGE = (1/Nr) sum P(i,j)*j^2/(i+1)^2. |
| `func (rl *RunLengthMatrix) LowGrayLevelRunEmphasis() float64` | LowGrayLevelRunEmphasis returns LGRE = (1/Nr) sum P(i,j)/(i+1)^2, emphasising runs of low gray level. |
| `func (rl *RunLengthMatrix) MaxRun() int` | MaxRun returns the longest run length the matrix can represent (its column count). |
| `func (rl *RunLengthMatrix) RunLengthNonUniformity() float64` | RunLengthNonUniformity returns RLN = (1/Nr) sum_j (sum_i P(i,j))^2, low when run lengths are evenly distributed and high when one length dominates. |
| `func (rl *RunLengthMatrix) RunPercentage() float64` | RunPercentage returns RP = Nr/Np, the ratio of runs to pixels. |
| `func (rl *RunLengthMatrix) ShortRunEmphasis() float64` | ShortRunEmphasis returns SRE = (1/Nr) sum P(i,j)/j^2, which is large for textures dominated by short runs (fine texture). |
| `func (rl *RunLengthMatrix) ShortRunHighGrayLevelEmphasis() float64` | ShortRunHighGrayLevelEmphasis returns SRHGE = (1/Nr) sum P(i,j)*(i+1)^2/j^2. |
| `func (rl *RunLengthMatrix) ShortRunLowGrayLevelEmphasis() float64` | ShortRunLowGrayLevelEmphasis returns SRLGE = (1/Nr) sum P(i,j)/((i+1)^2 j^2). |
| `func (rl *RunLengthMatrix) TotalRuns() float64` | TotalRuns returns Nr, the total number of runs tallied in the matrix. |

</details>

<details>
<summary><code>TamuraFeatureSet</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func TamuraFeatures(img *cv.Mat) TamuraFeatureSet` | TamuraFeatures computes all Tamura features of img with default parameters (coarseness kmax=5, directionality nbins=16 threshold=12) and returns them… |

</details>

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/texture).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
