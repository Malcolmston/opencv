# cudawarping — CPU-backed, API-compatible mirror of OpenCV's cudawarping module (the cv::cuda

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/cudawarping.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/cudawarping)

Package cudawarping is a CPU-backed, API-compatible mirror of OpenCV's
cudawarping module (the cv::cuda geometric image transformations) built on the
standard-library-only OpenCV port github.com/malcolmston/opencv, imported here
as cv.

Despite the name, this package contains no CUDA, no cgo and no GPU code. It
exists so that Go programs written against OpenCV's cudawarping surface —
the `GpuMat` container, a `Stream` handle, and warping calls that take flags,
border modes and streams — compile and run using only the Go standard
library and the root cv package. Every `GpuMat` holds its pixels in ordinary
host memory (a wrapped `cv.Mat`) and every operation is computed synchronously
on the CPU. `Upload` and `GpuMat.Download` are deep copies rather than
host↔device transfers, and `Stream` is an inert handle whose methods are
no-ops. Results match the CPU reference implementation in cv; they are not
bit-exact with a real CUDA build, and there is no performance benefit over
calling cv directly. Use this package for portability and drop-in source
compatibility, not for acceleration.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/cudawarping"
```

## Usage

This is the package's own `Example`, so it compiles and its output is
asserted on every `go test ./cudawarping/`.

```go
host := cv.NewMat(2, 2, 1)
	host.SetTo(7)

	g := cudawarping.Upload(host)
	out := g.Resize(image.Point{X: 4, Y: 4}, 0, 0, cudawarping.InterLinear, nil).Download()

	fmt.Printf("%dx%d value=%d\n", out.Rows, out.Cols, out.At(0, 0, 0))
```

```
4x4 value=7
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func BuildRotationMaps(angle, xShift, yShift float64, dsize image.Point, stream *Stream) (xmap, ymap *cv.FloatMat)` | BuildRotationMaps builds the coordinate maps that realise the cuda::rotate transform (rotation by angle degrees about the origin followed by a shift… |
| `func BuildWarpAffineMaps(m cv.AffineMatrix, inverse bool, dsize image.Point, stream *Stream) (xmap, ymap *cv.FloatMat)` | BuildWarpAffineMaps builds the pair of coordinate maps that `GpuMat.Remap` (or OpenCV's cuda::remap) needs to realise the affine transform m over an… |
| `func BuildWarpPerspectiveMaps(m cv.PerspectiveMatrix, inverse bool, dsize image.Point, stream *Stream) (xmap, ymap *cv.FloatMat)` | BuildWarpPerspectiveMaps builds the coordinate maps that realise the projective transform m over an output of size dsize, the perspective analogue of… |

### Types

| Type | What it is |
| --- | --- |
| `BorderMode` | BorderMode selects how samples outside the source image are produced, matching OpenCV's cv::BorderTypes numeric values. |
| `GpuMat` | GpuMat is a CPU-backed stand-in for OpenCV's cv::cuda::GpuMat. |
| `Interpolation` | Interpolation selects the resampling method used by the warp operations. |
| `Point2f` | Point2f is a sub-pixel image coordinate (x is the column, y is the row), used as the centre for the polar warps. |
| `RotateCode` | RotateCode selects one of the three lossless right-angle rotations for `GpuMat.Rotate90`. |
| `Stream` | Stream is a CPU-backed stand-in for cv::cuda::Stream, the handle that schedules asynchronous work on a CUDA stream. |

<details>
<summary><code>GpuMat</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewGpuMat(rows, cols, channels int) *GpuMat` | NewGpuMat allocates a zero-filled GpuMat with the given dimensions, mirroring the cv::cuda::GpuMat(rows, cols, type) constructor. |
| `func Upload(src *cv.Mat) *GpuMat` | Upload copies a host `cv.Mat` into a new GpuMat, mirroring cv::cuda::GpuMat::upload. |
| `func (g *GpuMat) Channels() int` | Channels returns the number of samples per pixel, or 0 for an empty GpuMat. |
| `func (g *GpuMat) Clone() *GpuMat` | Clone returns a deep copy of the GpuMat with its own backing storage, mirroring cv::cuda::GpuMat::clone. |
| `func (g *GpuMat) CopyMakeBorder(top, bottom, left, right int, borderMode BorderMode, borderValue float64, stream *Stream) *GpuMat` | CopyMakeBorder returns a copy of the GpuMat enlarged by top, bottom, left and right rows/columns of border, filled according to borderMode (with… |
| `func (g *GpuMat) Download() *cv.Mat` | Download copies the GpuMat back to a host `cv.Mat`, mirroring cv::cuda::GpuMat::download. |
| `func (g *GpuMat) Empty() bool` | Empty reports whether the GpuMat holds no samples, mirroring cv::cuda::GpuMat::empty. |
| `func (g *GpuMat) Flip(flipCode int, stream *Stream) *GpuMat` | Flip mirrors the GpuMat about one or both axes and mirrors cv::cuda::flip. |
| `func (g *GpuMat) LinearPolar(dsize image.Point, center Point2f, maxRadius float64, flags int, stream *Stream) *GpuMat` | LinearPolar remaps the GpuMat into (radius, angle) space about center with a linear radial axis, the convenience form of `GpuMat.WarpPolar` with… |
| `func (g *GpuMat) LogPolar(dsize image.Point, center Point2f, maxRadius float64, flags int, stream *Stream) *GpuMat` | LogPolar remaps the GpuMat into (log-radius, angle) space about center, the convenience form of `GpuMat.WarpPolar` with `WarpPolarLog`. |
| `func (g *GpuMat) PyrDown(stream *Stream) *GpuMat` | PyrDown blurs the GpuMat with the 5-tap binomial Gaussian kernel and drops every other row and column, halving each dimension (rounding up). |
| `func (g *GpuMat) PyrUp(stream *Stream) *GpuMat` | PyrUp doubles each dimension of the GpuMat by inserting zero rows and columns and smoothing with the 5-tap Gaussian kernel, the up-sampling step of a… |
| `func (g *GpuMat) Release()` | Release frees the GpuMat's backing store, mirroring cv::cuda::GpuMat::release. |
| `func (g *GpuMat) Remap(xmap, ymap *cv.FloatMat, interp Interpolation, borderMode BorderMode, borderValue float64, stream *Stream) *GpuMat` | Remap resamples the GpuMat at the per-pixel source coordinates given by xmap and ymap: destination pixel (x, y) is taken from the source at… |
| `func (g *GpuMat) Resize(dsize image.Point, fx, fy float64, interp Interpolation, stream *Stream) *GpuMat` | Resize scales the GpuMat to the destination size dsize (width dsize.X, height dsize.Y) using the chosen interpolation, mirroring cv::cuda::resize. |
| `func (g *GpuMat) ResizeScale(fx, fy float64, interp Interpolation, stream *Stream) *GpuMat` | ResizeScale scales the GpuMat by the independent factors fx (horizontal) and fy (vertical), a thin wrapper over `GpuMat.Resize` with a zero dsize. |
| `func (g *GpuMat) Rotate(dsize image.Point, angle, xShift, yShift float64, interp Interpolation, stream *Stream) *GpuMat` | Rotate rotates the GpuMat by angle degrees about the origin, then shifts the result by (xShift, yShift), producing an output of size dsize. |
| `func (g *GpuMat) Rotate90(code RotateCode, stream *Stream) *GpuMat` | Rotate90 performs a lossless 90/180/270-degree rotation of the GpuMat, delegating to `cv.Rotate`. |
| `func (g *GpuMat) Size() (rows, cols int)` | Size returns the GpuMat dimensions as (rows, cols), matching `cv.Mat.Size`. |
| `func (g *GpuMat) Transpose(stream *Stream) *GpuMat` | Transpose swaps the rows and columns of the GpuMat, returning a result of shape cols×rows. |
| `func (g *GpuMat) Upload(src *cv.Mat)` | Upload replaces the contents of g with a deep copy of the host matrix src, mirroring cv::cuda::GpuMat::upload on an existing GpuMat. |
| `func (g *GpuMat) WarpAffine(m cv.AffineMatrix, dsize image.Point, flags int, borderMode BorderMode, borderValue float64, stream *Stream) *GpuMat` | WarpAffine applies the 2×3 affine transform m to the GpuMat, producing an output of size dsize (width dsize.X, height dsize.Y). |
| `func (g *GpuMat) WarpPerspective(m cv.PerspectiveMatrix, dsize image.Point, flags int, borderMode BorderMode, borderValue float64, stream *Stream) *GpuMat` | WarpPerspective applies the 3×3 projective transform m to the GpuMat, producing an output of size dsize, the perspective analogue of… |
| `func (g *GpuMat) WarpPolar(dsize image.Point, center Point2f, maxRadius float64, flags int, stream *Stream) *GpuMat` | WarpPolar remaps the GpuMat between Cartesian and polar coordinates about center, mirroring cv::warpPolar / cv::cuda's polar helpers. |

</details>

<details>
<summary><code>Stream</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewStream() *Stream` | NewStream returns a new no-op `Stream`. |
| `func (s *Stream) QueryIfComplete() bool` | QueryIfComplete always reports true, since work is never outstanding in the CPU port. |
| `func (s *Stream) WaitForCompletion()` | WaitForCompletion returns immediately: in this CPU port all work has already finished synchronously by the time a Stream would be waited on. |

</details>

### Constants

`WarpFillOutliers`, `WarpInverseMap`, `WarpPolarLinear`, `WarpPolarLog`, `Rotate90CW`, `Rotate180`, `Rotate90CCW`

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/cudawarping).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
