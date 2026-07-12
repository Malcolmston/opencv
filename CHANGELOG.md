# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-07-12
### Added
- Initial release — a standard-library-only Go port of a useful subset of
  OpenCV (`cv2`) for image processing and computer vision, with zero
  dependencies.
- `Mat` core: dense row-major `[]uint8` matrix with `Clone`, `Region`,
  `CopyTo`, `Split`/`Merge`, `SetTo`, element and pixel accessors, and
  `FromImage` / `ToImage` bridges to the standard library.
- PNG + JPEG I/O via `ImRead`/`ImWrite` and in-memory `IMDecode`/`IMEncode`.
- Color conversions (`CvtColor`: RGB↔Gray, RGB↔BGR, RGB↔HSV) and `InRange`.
- Filtering and convolution: `Filter2D`, `Blur`, `BoxFilter`, `GaussianBlur`,
  `MedianBlur`, `Sobel`, `Scharr`, `Laplacian`.
- Thresholding (`Threshold` incl. Otsu, `AdaptiveThreshold`), morphology
  (`Erode`, `Dilate`, `MorphologyEx`, `GetStructuringElement`), and geometric
  transforms (`Resize`, `Flip`, `Rotate`, `Transpose`, `WarpAffine`,
  `GetRotationMatrix2D`).
- Edge detection (`Canny`) and template matching (`MatchTemplate` / `MinMaxLoc`).
- Drawing primitives (`Line`, `Rectangle`, `Circle`, `Ellipse`, `Polylines`,
  `FillPoly`) and bitmap-font text (`PutText`), plus histograms (`CalcHist`,
  `EqualizeHist`).
- CI: gofmt · vet · build gate a `-race` + coverage run, plus golangci-lint v2,
  govulncheck, cross-compile, dependency review, and VERSION-driven releases.

[Unreleased]: https://github.com/malcolmston/opencv/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/malcolmston/opencv/releases/tag/v0.1.0
