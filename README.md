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

## Modules

Beyond the root `cv` package, the library ships **57 packages** that mirror the
layout of OpenCV's main, `contrib` and (via CPU-backed shims) `cuda` modules —
around **5,300 exported functions** in total. Each imports only the root `cv`
package and the Go standard library — no cgo, no third-party dependencies — and
each carries full godoc, deterministic tests and runnable examples (80–100 %
statement coverage). Import the ones you need:

```go
import (
    cv "github.com/malcolmston/opencv"
    "github.com/malcolmston/opencv/features2d"
    "github.com/malcolmston/opencv/calib3d"
)
```

- **2D features** — `features2d` (ORB/BRIEF/SIFT/KAZE/AKAZE/BFMatcher/FLANN/BOW),
  `xfeatures2d` (FREAK/DAISY/LATCH/SURF-lite/BEBLID + matchGMS), `flann`
  (KD-tree/k-means/LSH/hierarchical/autotuned ANN), `linedescriptor` (LSD/EDLines
  + LBD + LSH matcher).
- **Geometry & 3D** — `calib3d` (calibrate/stereo-calibrate/rectify/solvePnP/
  essential/fundamental/homography/chessboard), `stereo` (BM/SGBM/8-path-SGM/
  census/WLS), `rgbd` (depth→3D, ICP, RGBD odometry, TSDF), `surface_matching`
  (PPF + KD-tree ICP), `structured_light`, `phase_unwrapping`, `ccalib`
  (omnidirectional camera), `rapid` (3D pose tracking), `imgprocx` (affine
  estimate, Gabor, log-polar, EMD).
- **Motion & tracking** — `video` (LK/Farnebäck/Kalman/ECC/DIS/stabilizer),
  `optflow` (TV-L1/DeepFlow/PCAFlow/RLOF/SimpleFlow), `tracking` (MOSSE/DCF/
  KCF-HOG/CSRT/MIL/Boosting/TLD/MultiTracker), `bgsegm` (MOG/MOG2/KNN/CNT/LSBP/
  GSOC), `videostab` (global motion + trajectory smoothing + inpaint/deblur).
- **Detection & recognition** — `objdetect` (HOG/cascade/QR + NMS), `aruco`
  (markers + ChArUco boards/diamonds), `face` (Eigen/Fisher/LBPH + Facemark/
  MACE/BIF), `barcode` (QR v1–10 + 8 1-D symbologies), `datamatrix` (full ECC200
  codec), `text` (MSER/ER/SWT/OCR + beam search), `dnn` (feed-forward CNN
  inference), `saliency` (spectral/Itti-Koch/GMR/BMS + objectness), `xobjdetect`
  (WaldBoost).
- **Photo & imaging** — `photo` (denoise/inpaint/seamless-clone/stylization),
  `hdr` (Debevec/Robertson/Mertens + AlignMTB + tonemap), `xphoto` (white
  balance/BM3D/dehaze/FSR), `intensity` (gamma/Retinex/AGCWD/BIMEF), `fuzzy`
  (F-transform), `bioinspired` (retina + transient-areas), `dnn_superres`
  (LapSRN/ESPCN/NEDI/DCCI, classical).
- **Segmentation, shape & stitching** — `segmentation` (watershed/GrabCut/
  Felzenszwalb/selective-search/livewire), `shape` (shape-context/TPS/Hausdorff/
  Hu moments), `ximgproc` (domain-transform/FGS/EdgeBoxes/SLIC/LSC), `stitching`
  (warpers/exposure/seams/bundle-adjust/pipeline), `hfs`.
- **Analysis & viz** — `ml` (KNN/SVM/RTrees/Boost/MLP/GMM/k-means + metrics),
  `quality` (PSNR/SSIM/VIF/FSIM/NIQE/…), `imghash`, `plot` (14 chart types + 21
  colormaps), `videoio` (GIF/APNG/MJPEG-AVI/sequences), `mcc` (ColorChecker +
  CIEDE2000 + CCM), `gapi` (lazy computation-graph G-API).

### CUDA-family packages (CPU-backed)

A `cuda*` family — `cudaarithm`, `cudaimgproc`, `cudafilters`, `cudawarping`,
`cudafeatures2d`, `cudabgsegm`, `cudaobjdetect`, `cudaoptflow`, `cudastereo`,
`cudacodec`, `cudacore`, `cudalegacy` — mirrors the **API shape** of OpenCV's
GPU modules so code ports naturally: same `GpuMat` / `Stream` vocabulary,
`Upload`/`Download` calls, and function signatures. They are **CPU-backed and
cgo-free** — `GpuMat` wraps an ordinary host `Mat` and `Stream` is a no-op, so
you get API parity, *not* hardware acceleration. Every function is documented as
such.

## Scope & limits

- **CV_8U only.** Samples are 8-bit unsigned; there is no floating-point or
  higher-bit-depth `Mat`. (Intermediate results such as `MatchTemplate` scores
  use a `FloatMat`.)
- **RGB, not BGR.** By Go convention three-channel data is treated as RGB
  (matching the `image` package), not OpenCV's native BGR. Use `CvtColor` with
  `ColorRGB2BGR` when you need to interoperate with BGR-oriented code or data.
- **Approximations, not trained models.** The module subpackages implement the
  classical algorithms directly; where OpenCV ships pre-trained weights or model
  files (deep SISR networks, learning-based white balance, DNN detectors, trained
  ER/BING classifiers) the Go ports use faithful weight-free approximations and
  say so in each package's `doc.go`. Every subpackage documents what it covers
  versus what it defers.

## Documentation

- Full API reference on [pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv).
- Docs site: <https://malcolmston.github.io/opencv/>.

## License

MIT
