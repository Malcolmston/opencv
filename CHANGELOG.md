# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-07-12
### Added
- Contours & shape analysis: `FindContours` (Suzuki border following with
  RETR_EXTERNAL/LIST/TREE retrieval, CHAIN_APPROX_NONE/SIMPLE and a hierarchy),
  `DrawContours`, `ContourArea`, `ArcLength`, `BoundingRect`, `MinAreaRect`
  (rotating calipers), `ConvexHull` (monotone chain), `ApproxPolyDP`
  (Douglas–Peucker) and `ImageMoments` (spatial/central/normalised + centroid).
- Connected components: `ConnectedComponents` and
  `ConnectedComponentsWithStats` (two-pass union-find, 4/8 connectivity).
- Feature detection: `CornerHarris`, `GoodFeaturesToTrack` (Shi–Tomasi),
  `HoughLines`, `HoughLinesP`, `HoughCircles` and `FASTCorners`.
- Projective geometry: `GetPerspectiveTransform` + `WarpPerspective`, `Remap`,
  the `PyrDown`/`PyrUp` Gaussian pyramid and `DistanceTransform`.
- Colour spaces: `CvtColor` now converts RGB↔Lab, RGB↔YCrCb and RGB↔HLS.
- Filtering: edge-preserving `BilateralFilter` and separable `Filter2DSep`.
- Arithmetic & logic: `Add`, `Subtract`, `AbsDiff`, `AddWeighted`, `Multiply`,
  `Divide`, `BitwiseAnd`/`Or`/`Xor`/`Not`, `Min`, `Max`, `Normalize` and
  `ConvertScaleAbs` — element-wise with saturation.
- Histograms: `CalcBackProject`, `CompareHist` and `CLAHE`.
- Examples now demonstrate contour detection and a perspective warp.

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

[Unreleased]: https://github.com/malcolmston/opencv/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/malcolmston/opencv/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/malcolmston/opencv/releases/tag/v0.1.0
