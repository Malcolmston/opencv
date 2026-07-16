# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.0] - 2026-07-16
### Added
- **CUDA-family packages (CPU-backed API mirrors).** Twelve new packages —
  `cudaarithm`, `cudaimgproc`, `cudafilters`, `cudawarping`, `cudafeatures2d`,
  `cudabgsegm`, `cudaobjdetect`, `cudaoptflow`, `cudastereo`, `cudacodec`,
  `cudacore`, `cudalegacy` — mirror the API shape of OpenCV's GPU modules with a
  host-backed `GpuMat`, no-op `Stream`, and `Upload`/`Download` calls, delegating
  the actual computation to the root `cv` package and sibling modules. They are
  pure Go and cgo-free: API parity, not hardware acceleration, and every package
  documents this honestly.
- **`gapi`** — a pure-Go port of OpenCV's G-API: a lazy computation graph
  (`GMat`/`GComputation`/`GCompiled` + custom-kernel packages) with 38 core and
  imgproc graph operations that execute bit-identically to the eager pipeline.
- **New contrib modules** — `ccalib` (omnidirectional/fisheye camera model +
  custom calibration pattern), `xobjdetect` (WaldBoost detector over
  integral-channel features), `hfs` (hierarchical feature-selection
  segmentation), `rapid` (RAPID 3D object-pose tracking), and `videostab`
  (global motion estimation, trajectory smoothing, motion inpainting and
  deblurring).
- **Root `cv` deepening** — 64 additional core and imgproc functions: dense
  linear algebra (`Invert`, `Solve`, `Determinant`, `Eigen`, `SVDecomp`, `Gemm`,
  `PCACompute`/`Project`/`BackProject`, `Mahalanobis`, `CalcCovarMatrix`), array
  utilities (`Reduce`, `Repeat`, `Sort`, `MinMaxIdx`, `Transform`,
  `ExtractChannel`/`InsertChannel`/`MixChannels`), element-wise math (`Exp`,
  `Log`, `Pow`, `Sqrt`, `Magnitude`, `Phase`, `CartToPolar`, `PolarToCart`),
  signal routines (`DFT`/`IDFT`, `DCT`/`IDCT`, `MulSpectrums`,
  `CreateHanningWindow`, `PhaseCorrelate`), extended drawing (`DrawMarker`,
  `ArrowedLine`, `GetTextSize`, `FillConvexPoly`, `BoxPoints`) and geometry
  predicates (`PointPolygonTest`, `IsContourConvex`, `MinEnclosingCircle`,
  `FitLine`, `MatchShapes`, `HuMoments`).
- Documentation: corrected the root package overview (SIFT/ORB, calib3d, DNN and
  video are now implemented, not out of scope) and added a Subpackages map;
  refreshed the README Modules section and the docs-site content to cover all
  57 packages.

## [0.4.0] - 2026-07-12
### Added
- **~650 new functions deepening all 40 module subpackages toward OpenCV
  (`cv2`) parity**, delivered in four waves. Every addition is genuinely
  working (no stubs), imports only the root `cv` package + the Go standard
  library, and ships full godoc, deterministic tests and runnable examples
  (statement coverage 80–97%; golangci-lint clean). Where OpenCV relies on
  trained model files, the ports implement the underlying classical algorithm
  and document the approximation.
- Geometry/3D: `calib3d` (CalibrateCamera/StereoCalibrate/StereoRectify/
  SolvePnP(Ransac)/FindEssentialMat/RecoverPose/chessboard), `stereo` (8-path
  SGM/HH, census, WLS, quasi-dense), `rgbd` (point-to-plane & colored ICP,
  RGBD/ICP odometry, TSDF volume, FALS/LINEMOD/SRI normals), `surface_matching`
  (KD-tree ICP, point-to-plane, multi-instance PPF, PLY I/O), `structured_light`
  (multi-frequency/heterodyne unwrap, FTP, DLT triangulation, stereo decode),
  `phase_unwrapping` (Goldstein branch-cut, DCT-Poisson/PCG least-squares,
  Flynn, quality-guided, temporal).
- 2D features: `features2d` (SIFT/KAZE/AKAZE/FLANN matcher/BOW), `xfeatures2d`
  (FREAK/DAISY/LATCH/LUCID/SURF-lite/BEBLID + matchGMS/LOGOS), `linedescriptor`
  (multi-octave LBD, EDLines, LSH matcher), `flann` (hierarchical/composite/
  autotuned indices, more distances).
- Detection/recognition: `objdetect` (HOG/cascade DetectMultiScale, grouping,
  tracking), `aruco` (ChArUco boards/diamonds/board-pose/calibration), `face`
  (Facemark/MACE/BIF/HAAR + persistence), `barcode` (QR v1–10 all ECC levels +
  8 more 1D symbologies), `datamatrix` (C40/Text/X12/EDIFACT/Base256, rectangular
  & large sizes, structured-append/ECI/GS1), `text` (Neumann–Matas ER NM1/NM2,
  grouping, SWT, OCR template, beam search), `dnn` (17 more layers + NMSBoxes),
  `saliency` (Itti-Koch/MBD/frequency-tuned/context-aware/GMR/BMS/HC/RC + eval).
- Motion/tracking: `video` (ECC alignment, DIS flow, MeanShift/CamShift, PyrLK
  tracker, stabilizer, MOG2/KNN), `optflow` (TV-L1/DeepFlow/PCAFlow/RLOF/
  SimpleFlow + .flo I/O), `tracking` (FFT-based MOSSE/DCF/KCF-HOG/CSRT + MIL/
  Boosting/TLD + MultiTracker), `bgsegm` (MOG/CNT/LSBP/GSOC + shadow).
- Photo/imaging: `photo` (domain-transform/TV-L1/Poisson-editing/sketch/oil/
  cartoon), `hdr` (AlignMTB, MergeRobertson, TonemapDurand/Mantiuk, PFM/RGBE
  I/O), `xphoto` (DCT/BM3D-Wiener/dehaze/FSR/TonemapDurand/color-constancy),
  `intensity` (Retinex SSR/MSR/MSRCR, AGCWD, tone curve, WLS-BIMEF), `fuzzy`
  (degree-1 F-transform, multi-step inpaint, fast variants), `bioinspired`
  (transient-areas segmentation, Bayer mosaic/demosaic, log-polar, param I/O),
  `dnn_superres` (LapSRN/ESPCN/NEDI/DCCI/IBP + SSIM/benchmark).
- Analysis/misc: `ml` (RandomForest/AdaBoost/MLP/GMM/kernel-SVM + metrics +
  persistence), `segmentation` (Felzenszwalb/selective-search/multi-Otsu/exact
  DT/RAG/livewire/SLIC), `imgprocx` (kernel builders/spatial-gradient/accumulate/
  EMD/floodfill/tilted-integral), `quality` (VIF/FSIM/VSI/NIQE/PIQE/IW-SSIM/
  CW-SSIM), `ximgproc` (domain-transform/FGS/adaptive-manifold/weighted-median/
  Deriche/EdgeBoxes/LSC), `shape` (shape-context+Hungarian+TPS, Hausdorff,
  EMD-L1, robust fit), `mcc` (CIEDE2000/CIE94/CMC, chromatic adaptation, LCh/xyY,
  140-patch DigitalSG, poly/root-poly/WLS CCM), `imghash` (Haar-DWT WaveletHash,
  72-bit Marr–Hildreth, peak-cross-correlation), `stitching` (cyl/spherical
  warpers, exposure compensation, DP/graph-cut seams, LM bundle adjustment,
  wave-correct, timelapse, pipeline), `plot` (stem/step/area/box/violin/pie/
  heatmap/contour/multi-series + 13 colormaps), `videoio` (APNG, MJPEG-in-AVI,
  image-sequence, CAP_PROP model, adaptive GIF palette).

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
