# opencv

[![Go Test](https://github.com/Malcolmston/opencv/actions/workflows/go-test.yml/badge.svg)](https://github.com/Malcolmston/opencv/actions/workflows/go-test.yml)
[![Go Lint](https://github.com/Malcolmston/opencv/actions/workflows/go-lint.yml/badge.svg)](https://github.com/Malcolmston/opencv/actions/workflows/go-lint.yml)
[![Go Vuln](https://github.com/Malcolmston/opencv/actions/workflows/go-vuln.yml/badge.svg)](https://github.com/Malcolmston/opencv/actions/workflows/go-vuln.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv.svg)](https://pkg.go.dev/github.com/malcolmston/opencv)
[![Go Report Card](https://goreportcard.com/badge/github.com/malcolmston/opencv)](https://goreportcard.com/report/github.com/malcolmston/opencv)
[![Go Version](https://img.shields.io/github/go-mod/go-version/Malcolmston/opencv)](go.mod)
[![Release](https://img.shields.io/github/v/release/Malcolmston/opencv?sort=semver)](https://github.com/Malcolmston/opencv/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A standard-library-only Go port of OpenCV — image processing & computer vision with zero dependencies.

## What it is

`cv` is a from-scratch, standard-library-only port of a useful subset of Python's
OpenCV (`cv2`), focused on classic image-processing and computer-vision
primitives. It is written entirely against the Go standard library — `image`,
`image/color`, `image/png`, `image/jpeg`, `math` and friends — with no cgo and no
third-party dependencies, so it builds and runs anywhere the Go toolchain does.

The central data structure is `Mat`, a dense row-major matrix of 8-bit unsigned
samples backed by a flat `[]uint8`. One-channel (grayscale) and three-channel
(RGB) images are the common cases; convert to and from the standard library with
`FromImage` / `Mat.ToImage`, and read or write PNG/JPEG with `ImRead` / `ImWrite`.

## Installation

```sh
go get github.com/malcolmston/opencv
```

```go
import cv "github.com/malcolmston/opencv"
```

The module path is `github.com/malcolmston/opencv`; the package name is `cv`.

## Quick start

Load an image, convert it to grayscale, blur it, and run a Canny edge detector:

```go
package main

import (
	"log"

	cv "github.com/malcolmston/opencv"
)

func main() {
	img, err := cv.ImRead("in.png")
	if err != nil {
		log.Fatal(err)
	}

	gray := cv.CvtColor(img, cv.ColorRGB2Gray)
	blur := cv.GaussianBlur(gray, 5, 1.4)
	edges := cv.Canny(blur, 50, 150)

	if err := cv.ImWrite("edges.png", edges); err != nil {
		log.Fatal(err)
	}
}
```

## Features

- **Mat core** — dense row-major `[]uint8` matrix (`NewMat`, `Clone`, `Region`,
  `CopyTo`, `Split`/`Merge`, `SetTo`, `At`/`Set`, `AtPixel`/`SetPixel`, `Size`,
  `Total`, `Empty`) and standard-library bridges `FromImage` / `Mat.ToImage`.
- **I/O (PNG + JPEG)** — `ImRead` / `ImWrite` for files and `IMDecode` /
  `IMEncode` for in-memory buffers, via the standard-library codecs.
- **Color conversions** — `CvtColor` with RGB↔Gray, RGB↔BGR, RGB↔HSV, RGB↔Lab,
  RGB↔YCrCb and RGB↔HLS (`ColorRGB2Gray`, `ColorGray2RGB`, `ColorRGB2BGR`,
  `ColorBGR2RGB`, `ColorBGR2Gray`, `ColorRGB2HSV`, `ColorHSV2RGB`, `ColorRGB2Lab`,
  `ColorLab2RGB`, `ColorRGB2YCrCb`, `ColorYCrCb2RGB`, `ColorRGB2HLS`,
  `ColorHLS2RGB`), plus `InRange` masking.
- **Filtering / convolution** — generic `Filter2D` (`Kernel` / `NewKernel`),
  separable `Filter2DSep`, and the built-ins: `Blur`, `BoxFilter`,
  `GaussianBlur` (`GaussianKernel1D`), `MedianBlur`, edge-preserving
  `BilateralFilter`, `Sobel`, `Scharr` and `Laplacian`.
- **Arithmetic & logic** — element-wise `Add`, `Subtract`, `AbsDiff`,
  `AddWeighted`, `Multiply`, `Divide`, `BitwiseAnd`/`Or`/`Xor`/`Not`, `Min`,
  `Max`, `Normalize` and `ConvertScaleAbs`, all with saturation.
- **Thresholding** — fixed and `Otsu` levels through `Threshold`, plus
  `AdaptiveThreshold` (mean and Gaussian).
- **Morphology** — `Erode`, `Dilate` and `MorphologyEx` (open, close, gradient,
  tophat, blackhat) over structuring elements from `GetStructuringElement`.
- **Geometric transforms** — `Resize` (nearest / bilinear), `Flip`, `Rotate`,
  `Transpose`, affine warping via `WarpAffine` / `GetRotationMatrix2D`,
  projective warping via `GetPerspectiveTransform` / `WarpPerspective`, `Remap`,
  the `PyrDown` / `PyrUp` Gaussian pyramid and `DistanceTransform`.
- **Contours & shape** — Suzuki-style `FindContours` (RETR_EXTERNAL/LIST/TREE,
  CHAIN_APPROX_NONE/SIMPLE with hierarchy), `DrawContours`, `ContourArea`,
  `ArcLength`, `BoundingRect`, `MinAreaRect`, `ConvexHull`, `ApproxPolyDP` and
  `ImageMoments`.
- **Connected components** — `ConnectedComponents` and
  `ConnectedComponentsWithStats` (union-find, 4/8 connectivity).
- **Feature detection** — `CornerHarris`, `GoodFeaturesToTrack` (Shi-Tomasi),
  `HoughLines`, `HoughLinesP`, `HoughCircles` and `FASTCorners`.
- **Edges & template matching** — a full `Canny` pipeline and `MatchTemplate`
  with `MinMaxLoc`.
- **Drawing & text** — `Line`, `Rectangle`, `Circle`, `Ellipse`, `Polylines`,
  `FillPoly`, and `PutText` rendered with a built-in bitmap font.
- **Histograms** — `CalcHist`, `EqualizeHist`, `CalcBackProject`, `CompareHist`
  and contrast-limited adaptive equalisation (`CLAHE`).

## Scope & limits

- **CV_8U only.** Samples are 8-bit unsigned; there is no floating-point or
  higher-bit-depth `Mat`. (Intermediate results such as `MatchTemplate` scores
  use a `FloatMat`.)
- **RGB, not BGR.** By Go convention three-channel data is treated as RGB
  (matching the `image` package), not OpenCV's native BGR. Use `CvtColor` with
  `ColorRGB2BGR` when you need to interoperate with BGR-oriented code or data.
- **Deferred.** Heavyweight machine-vision machinery is intentionally out of
  scope: dense feature descriptors and matching (SIFT/ORB/BRIEF), camera
  calibration and stereo (calib3d), DNN inference, optical flow and video I/O.

## Documentation

- Full API reference on [pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv).
- Docs site: <https://malcolmston.github.io/opencv/>.

## License

MIT
