# moments2 — provides image moments and shape descriptors for the github.com/malcolmston/opencv

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/moments2.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/moments2)

Package moments2 provides image moments and shape descriptors for the
github.com/malcolmston/opencv library.

The package operates on the library's existing image type, `cv.Mat`, and its
geometry types `cv.Point` and `cv.Point2f`; it does not introduce a competing
image representation. Functions accept single-channel Mats (grayscale or
binary masks) and polygonal contours ([]cv.Point) as produced by the parent
package's contour extraction.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/moments2"
```

## Usage

This is the package's own `ExampleContourMoments`, so it compiles and its output is
asserted on every `go test ./moments2/`.

```go
m := moments2.ContourMoments(square(10))
	c := m.Centroid()
	fmt.Printf("area=%.0f centroid=(%.1f,%.1f)\n", m.Area(), c.X, c.Y)

	fmt.Printf("mu11=%.0f mu30=%.0f\n", m.Mu11, m.Mu30)
```

```
area=100 centroid=(5.0,5.0)
mu11=0 mu30=0
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func AspectRatio(contour []cv.Point) float64` | AspectRatio returns the ratio of the width to the height of a contour's upright bounding rectangle. |
| `func AxisLengths(m Moments) (major, minor float64)` | AxisLengths returns the lengths of the major and minor axes of the ellipse that has the same second central moments as the region described by m. |
| `func CentralMoment(src *cv.Mat, p, q int) float64` | CentralMoment returns the central moment of order (p, q), taken about the image centroid, for any non-negative p and q. |
| `func Circularity(area, perimeter float64) float64` | Circularity returns the normalized circularity 4*pi*area/perimeter^2 of a region, a value in (0, 1] where 1 is a perfect circle. |
| `func Compactness(area, perimeter float64) float64` | Compactness returns the isoperimetric compactness perimeter^2/area of a region. |
| `func ConvexHullIndices(contour []cv.Point) []int` | ConvexHullIndices returns the indices into contour of its convex hull vertices in counter-clockwise order (in image coordinates, where y grows… |
| `func Convexity(contour []cv.Point) float64` | Convexity returns the ratio of the convex-hull perimeter to the contour perimeter, a value in (0, 1] where 1 means the contour is convex. |
| `func ConvexityRatio(contour []cv.Point) float64` | ConvexityRatio returns a scalar convexity measure equal to one divided by one plus the total defect depth of a contour, normalized so that a convex… |
| `func Eccentricity(m Moments) float64` | Eccentricity returns the eccentricity of the equivalent ellipse of m, a value in [0, 1) where 0 is a perfect circle and values approaching 1 are… |
| `func Elongation(m Moments) float64` | Elongation returns the ratio of the major to the minor axis length of the equivalent ellipse of m, always at least 1. |
| `func EquivalentDiameter(area float64) float64` | EquivalentDiameter returns the diameter of the circle whose area equals the given region area, sqrt(4*area/pi). |
| `func Extent(contour []cv.Point) float64` | Extent returns the ratio of a contour's area to the area of its upright bounding rectangle, a value in (0, 1]. |
| `func FlusserI1(m Moments) float64` | FlusserI1 returns the first Flusser & Suk affine moment invariant, computed from the central moments of m and normalized by mu00^4: |
| `func FlusserI2(m Moments) float64` | FlusserI2 returns the second Flusser & Suk affine moment invariant, built from the third-order central moments and normalized by mu00^10. |
| `func FlusserI3(m Moments) float64` | FlusserI3 returns the third Flusser & Suk affine moment invariant, mixing second- and third-order central moments and normalized by mu00^7. |
| `func FlusserI4(m Moments) float64` | FlusserI4 returns the fourth Flusser & Suk affine moment invariant, the most complex of the classic set, normalized by mu00^11. |
| `func FlusserInvariants(m Moments) [4]float64` | FlusserInvariants returns the four classic Flusser & Suk affine moment invariants as an array [I1, I2, I3, I4]. |
| `func FormFactor(area, perimeter float64) float64` | FormFactor is an alias of `Circularity`: 4*pi*area/perimeter^2. |
| `func FourierDescriptorDistance(a, b []float64) float64` | FourierDescriptorDistance returns the Euclidean distance between two normalized Fourier descriptor vectors, comparing only the overlapping leading… |
| `func FourierDescriptors(contour []cv.Point2f) []complex128` | FourierDescriptors returns the discrete Fourier transform of a closed contour whose points are treated as complex numbers x+iy. |
| `func FourierMagnitudeSpectrum(fd []complex128) []float64` | FourierMagnitudeSpectrum returns the moduli of a sequence of Fourier descriptors, discarding phase. |
| `func HuMoments(m Moments) [7]float64` | HuMoments returns Hu's seven invariant moments computed from the normalized central moments of m. |
| `func IsContourConvex(contour []cv.Point) bool` | IsContourConvex reports whether a polygon is convex, that is whether every turn along its boundary has the same sign. |
| `func LegendreMoment(src *cv.Mat, p, q int) float64` | LegendreMoment computes the Legendre moment L_pq of a single-channel image. |
| `func LegendreMoments(src *cv.Mat, maxOrder int) [][]float64` | LegendreMoments returns all Legendre moments L_pq with p+q <= maxOrder as a dense (maxOrder+1) x (maxOrder+1) matrix; entries with p+q > maxOrder are… |
| `func LegendrePolynomial(n int, x float64) float64` | LegendrePolynomial evaluates the Legendre polynomial P_n(x) using the stable Bonnet recurrence. |
| `func LogHuMoments(m Moments) [7]float64` | LogHuMoments returns a sign-preserving log-magnitude transform of the seven Hu moments, -sign(h)*log10(\|h\|), which compresses their very wide… |
| `func MatchShapes(a, b Moments, method MatchMethod) float64` | MatchShapes returns a dissimilarity score between two shapes given by their moments, using their Hu invariants and the selected method. |
| `func MaxDiameter(contour []cv.Point) float64` | MaxDiameter returns the greatest distance between any two vertices of a contour (its Feret diameter). |
| `func NormalizedCentralMoment(src *cv.Mat, p, q int) float64` | NormalizedCentralMoment returns the normalized central moment of order (p, q), equal to mu_pq / M00^(1+(p+q)/2), which is invariant to uniform… |
| `func NormalizedFourierDescriptors(fd []complex128) []float64` | NormalizedFourierDescriptors returns a translation-, scale-, rotation- and start-point-invariant descriptor derived from fd. |
| `func Orientation(m Moments) float64` | Orientation returns the angle, in radians in the range (-pi/2, pi/2], of the major axis of the equivalent ellipse of m relative to the positive x… |
| `func PolygonArea(pts []cv.Point) float64` | PolygonArea returns the absolute area enclosed by a polygon using the shoelace formula. |
| `func PolygonCentroid(pts []cv.Point) cv.Point2f` | PolygonCentroid returns the area centroid of a polygon computed from its first moments. |
| `func PolygonPerimeter(pts []cv.Point) float64` | PolygonPerimeter returns the closed perimeter length of a polygon, the sum of the Euclidean edge lengths including the closing edge. |
| `func PseudoZernikeMoment(src *cv.Mat, n, m int) complex128` | PseudoZernikeMoment computes the complex pseudo-Zernike moment A_nm of a single-channel image over the inscribed unit disk. |
| `func PseudoZernikeRadial(n, m int, rho float64) float64` | PseudoZernikeRadial evaluates the pseudo-Zernike radial polynomial R_n^m(rho) for a radius rho in [0, 1]. |
| `func RadialPolynomial(n, m int, rho float64) float64` | RadialPolynomial evaluates the Zernike radial polynomial R_n^m(rho) for a radius rho in [0, 1]. |
| `func RawMoment(src *cv.Mat, p, q int) float64` | RawMoment returns the spatial moment of order (p, q), sum of value*x^p*y^q over all pixels, for any non-negative p and q. |
| `func ReconstructContour(fd []complex128, numPoints, numDescriptors int) []cv.Point2f` | ReconstructContour approximates a contour from its Fourier descriptors, keeping only the numDescriptors lowest-frequency coefficients (paired from… |
| `func Rectangularity(contour []cv.Point) float64` | Rectangularity returns the ratio of a contour's area to the area of its minimum-area rotated bounding rectangle, a value in (0, 1] measuring how well… |
| `func ResampleContour(contour []cv.Point, n int) []cv.Point2f` | ResampleContour resamples a closed contour to exactly n points spaced equally along its arc length, returning the new points as `cv.Point2f`. |
| `func Roundness(area, maxDiameter float64) float64` | Roundness returns 4*area/(pi*maxDiameter^2), a compactness measure that, unlike circularity, is insensitive to boundary roughness because it uses the… |
| `func ShapeContextCost(a, b ShapeContextHistogram) float64` | ShapeContextCost returns the chi-square dissimilarity between two shape context histograms, the standard local matching cost, in the range [0, 1]. |
| `func ShapeContextMatchCost(a, b []ShapeContextHistogram) float64` | ShapeContextMatchCost returns the mean over all reference points of the best per-point `ShapeContextCost` between two shape context descriptors, a… |
| `func Solidity(contour []cv.Point) float64` | Solidity returns the ratio of a contour's area to the area of its convex hull, a value in (0, 1] where 1 means the shape is convex. |
| `func ZernikeMagnitude(src *cv.Mat, n, m int) float64` | ZernikeMagnitude returns the modulus of the Zernike moment A_nm, a rotation invariant shape feature. |
| `func ZernikeMoment(src *cv.Mat, n, m int) complex128` | ZernikeMoment computes the complex Zernike moment A_nm of a single-channel image over the unit disk inscribed in the image. |

### Types

| Type | What it is |
| --- | --- |
| `ConvexityDefect` | ConvexityDefect describes a concavity of a contour relative to its convex hull: the segment of the hull it lies under (from Start to End) and the… |
| `EllipseParams` | EllipseParams describes an ellipse by its centre, full axis lengths and the orientation of its major axis in radians. |
| `MatchMethod` | MatchMethod selects the dissimilarity formula used by `MatchShapes`. |
| `Moments` | Moments holds the spatial (raw), central and normalized central moments of an image or contour up to the third order, mirroring OpenCV's cv::Moments. |
| `ShapeContextHistogram` | ShapeContextHistogram is the log-polar histogram computed at one reference point of a shape. |
| `ZernikeCoefficient` | ZernikeCoefficient is a single complex Zernike moment A_nm together with its order n and repetition m. |

<details>
<summary><code>ConvexityDefect</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func ConvexityDefects(contour []cv.Point, hull []int, minDepth float64) []ConvexityDefect` | ConvexityDefects finds the concavities of a contour with respect to its convex hull. |

</details>

<details>
<summary><code>EllipseParams</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func FitEllipse(m Moments) EllipseParams` | FitEllipse returns the ellipse that has the same centroid and second central moments as the region described by m. |

</details>

<details>
<summary><code>Moments</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func ContourMoments(pts []cv.Point) Moments` | ContourMoments computes the moments of the region enclosed by a polygon using Green's theorem, matching OpenCV's contour-based cv::moments. |
| `func ImageMoments(src *cv.Mat) Moments` | ImageMoments computes the full set of moments of a single-channel image, weighting each pixel (x, y) by its sample value. |
| `func MaskMoments(src *cv.Mat) Moments` | MaskMoments computes moments treating every non-zero pixel as a unit of mass, ignoring the actual sample value. |
| `func (m Moments) Area() float64` | Area returns the zeroth spatial moment M00, which for a binary mask equals the number of foreground pixels and for a contour equals the enclosed area. |
| `func (m Moments) Centroid() cv.Point2f` | Centroid returns the mass-weighted centre (M10/M00, M01/M00) as a `cv.Point2f`. |

</details>

<details>
<summary><code>ShapeContextHistogram</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func ComputeShapeContext(points []cv.Point2f, radialBins, angularBins int, innerRadius, outerRadius float64) []ShapeContextHistogram` | ComputeShapeContext computes the shape context descriptor of a point set: for each reference point it builds a log-polar histogram of the relative… |

</details>

<details>
<summary><code>ZernikeCoefficient</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func ZernikeMoments(src *cv.Mat, maxOrder int) []ZernikeCoefficient` | ZernikeMoments computes every valid Zernike moment up to and including radial order maxOrder, returning them ordered by n then by m from 0 to n. |
| `func (z ZernikeCoefficient) Magnitude() float64` | Magnitude returns the modulus of the coefficient, which is invariant to rotation of the underlying image. |
| `func (z ZernikeCoefficient) Phase() float64` | Phase returns the argument of the coefficient in radians. |

</details>

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/moments2).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
