# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-07-12
### Added
- **40 new module subpackages**, each importing only the root `cv` package plus
  the Go standard library (no cgo, no third-party dependencies), mirroring the
  architecture of OpenCV's main and `contrib` modules. Every subpackage ships
  full godoc, a `doc.go` overview, deterministic tests on synthetic images and
  runnable `Example` functions (statement coverage 86–97%).
- 2D features & description: `features2d` (ORB/BRIEF/BFMatcher/ratio test),
  `xfeatures2d` (AGAST/BRISK/Star/SimpleBlob/HarrisLaplace),
  `linedescriptor` (LSD + LBD binary line descriptors/matcher).
- Geometry & 3D: `calib3d` (homography+RANSAC, fundamental matrix, Rodrigues,
  solvePnP, triangulation, undistort), `stereo` (BM/SGBM + reproject-to-3D),
  `rgbd` (depth→3D, normals, plane segmentation, ICP, voxel downsample),
  `surface_matching` (PPF + ICP), `calib3d`-adjacent `imgprocx`
  (affine estimation, integral image, Gabor kernels, phase correlation,
  log/linear-polar).
- Motion & tracking: `video` (Lucas–Kanade/Farnebäck flow, Kalman),
  `optflow` (Horn–Schunck, DIS-lite, sparse-to-dense, flow colouring),
  `tracking` (template/KCF-lite/MedianFlow/MeanShift/CamShift),
  `bgsegm` (MOG2/KNN/GMG/running average).
- Detection & recognition: `objdetect` (HOG, cascade, QR-detect),
  `aruco` (marker generate/detect + pose), `face` (Eigen/Fisher/LBPH + LBP),
  `barcode` (QR/EAN-13/Code128 + Reed–Solomon), `datamatrix` (ECC200 codec),
  `text` (MSER scene-text detection + grouping), `dnn` (feed-forward CNN
  inference), `flann` (KD-tree/k-means/LSH ANN search),
  `saliency` (spectral-residual/fine-grained/motion/BING).
- Photo & computational imaging: `photo` (denoise/inpaint/edge-preserving/
  seamless clone), `hdr` (Debevec/Robertson calibration, Debevec/Mertens merge,
  Reinhard/Drago/Mantiuk tonemap), `xphoto` (white balance, oil painting,
  BM3D-lite, SHIFTMAP inpaint), `intensity` (gamma/log/BIMEF/histogram matching),
  `fuzzy` (F-transform filtering + inpainting), `bioinspired` (retina model +
  fast tone mapping), `dnn_superres` (bicubic/Lanczos/edge-directed upscaling).
- Structured light & phase: `structured_light` (Gray-code + phase-shift),
  `phase_unwrapping` (Herráez quality-guided unwrap).
- Segmentation & shape: `segmentation` (flood fill/watershed/GrabCut/mean-shift),
  `shape` (min-enclosing circle/triangle, fit line/ellipse, Hu moments,
  convexity defects), `ximgproc` (guided filter, thinning, SLIC, Niblack),
  `stitching` (feature-based panorama with feather/multi-band blending).
- Analysis & viz: `ml` (KNN/SVM/Bayes/logistic/tree/k-means),
  `quality` (MSE/PSNR/SSIM/MS-SSIM/GMSD/BRISQUE-lite), `imghash` (aHash/pHash/
  dHash/BlockMean/Marr–Hildreth/RadialVariance/ColorMoment), `plot` (line/scatter/
  bar/histogram + colormaps), `videoio` (GIF read/write via `image/gif`),
  `mcc` (Macbeth ColorChecker detection + colour-correction model).

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
