// Library content for the opencv documentation site. Mirrors the shape used by
// the malcolmston/go landing site's data.ts so the sibling sites stay in sync.
export interface Lib {
  id: string; name: string; icon: string; accent: string; pkg: string; node: string;
  repo: string; docs: string; tagline: string; blurb: string; tags: string[];
  features: string[]; node_code: string; go_code: string; integrate: string;
}

export const NODE_ACCENT = '#8cc84b';

export const OPENCV: Lib = {
  id:"opencv", name:"opencv", icon:'<i class="fa-solid fa-camera"></i>', accent:"#5cc8ff",
  pkg:"github.com/malcolmston/opencv", node:"opencv/opencv-python",
  repo:"https://github.com/malcolmston/opencv", docs:"https://malcolmston.github.io/opencv/",
  tagline:"Classic OpenCV image processing in Go.",
  blurb:"A from-scratch, standard-library-only Go port of a useful subset of Python's OpenCV (cv2), focused on "+
    "classic image processing and computer-vision primitives. Everything is built on the dense row-major "+
    "Mat type over image, image/color, image/png, image/jpeg and math — no cgo, no third-party dependencies. "+
    "You get the Mat core, PNG/JPEG I/O, colour conversions, filtering and convolution, thresholding, "+
    "morphology, geometric transforms, a full Canny pipeline, template matching, drawing and histograms — "+
    "an idiomatic, genuinely useful re-implementation of the cv2 essentials. On top of that root package "+
    "sit 57 packages (~5,300 functions) mirroring OpenCV's main, contrib and — via CPU-backed shims — cuda "+
    "modules: features2d/xfeatures2d, calib3d/stereo/rgbd, dnn, tracking/optflow/video, face/aruco/barcode, "+
    "photo/hdr/xphoto, stitching, ml, a lazy-graph G-API, and a full cuda* family that ports OpenCV's GPU "+
    "API to the CPU. Each imports only cv and the standard library. The import path is "+
    "github.com/malcolmston/opencv, but the package is named cv.",
  tags:["Mat core","PNG/JPEG I/O","colour conversion","convolution","Otsu threshold","morphology","warpAffine","Canny","57 packages","features2d","calib3d","dnn","tracking","G-API","cuda* (CPU)","stitching"],
  features:[
    "<code>Mat</code> core — a dense row-major matrix of 8-bit samples with <code>FromImage</code>/<code>ToImage</code> stdlib bridges",
    "PNG + JPEG I/O via <code>ImRead</code>, <code>ImWrite</code>, <code>IMDecode</code> and <code>IMEncode</code>",
    "Colour conversions through <code>CvtColor</code> (RGB↔Gray, RGB↔BGR, RGB↔HSV and more)",
    "Filtering &amp; convolution — <code>Filter2D</code>, <code>Blur</code>, <code>GaussianBlur</code>, <code>MedianBlur</code>, <code>Sobel</code>, <code>Scharr</code>, <code>Laplacian</code>",
    "Thresholding with automatic <code>Otsu</code> levels plus <code>AdaptiveThreshold</code>",
    "Morphology — <code>Erode</code>, <code>Dilate</code>, <code>MorphologyEx</code> over <code>GetStructuringElement</code> kernels",
    "Geometric transforms — <code>Resize</code>, <code>Flip</code>, <code>Rotate</code>, <code>Transpose</code>, <code>WarpAffine</code> + <code>GetRotationMatrix2D</code>",
    "Edges &amp; matching — a full <code>Canny</code> pipeline and <code>MatchTemplate</code> with <code>MinMaxLoc</code>",
    "Drawing &amp; text — <code>Line</code>, <code>Rectangle</code>, <code>Circle</code>, <code>Ellipse</code>, <code>Polylines</code>, <code>FillPoly</code>, <code>PutText</code>",
    "Histograms — <code>CalcHist</code> and <code>EqualizeHist</code>",
    "<b>57 packages · ~5,300 functions</b> mirroring OpenCV main + contrib + cuda, each importing only <code>cv</code> and the stdlib",
    "2D features — <code>features2d</code> (ORB/SIFT/KAZE/AKAZE/matcher/BOW), <code>xfeatures2d</code> (FREAK/DAISY/SURF-lite), <code>flann</code>, <code>linedescriptor</code>",
    "Geometry &amp; 3D — <code>calib3d</code> (calibrate/stereo/solvePnP/essential), <code>stereo</code> (8-path SGM), <code>rgbd</code> (ICP/odometry/TSDF), <code>surface_matching</code>, <code>ccalib</code>, <code>rapid</code>",
    "Motion &amp; tracking — <code>video</code> (LK/Kalman/ECC/DIS), <code>optflow</code> (TV-L1/DeepFlow/RLOF), <code>tracking</code> (MOSSE/CSRT/TLD), <code>bgsegm</code>, <code>videostab</code>",
    "Detection &amp; recognition — <code>objdetect</code>, <code>aruco</code> (+ChArUco), <code>face</code> (+Facemark/MACE), <code>barcode</code> (QR v1–10), <code>datamatrix</code>, <code>text</code> (ER/SWT/OCR), <code>dnn</code>, <code>saliency</code>, <code>xobjdetect</code>",
    "Photo &amp; imaging — <code>photo</code>, <code>hdr</code> (AlignMTB/tonemap), <code>xphoto</code> (BM3D/dehaze), <code>intensity</code> (Retinex/BIMEF), <code>fuzzy</code>, <code>bioinspired</code>, <code>dnn_superres</code>",
    "Segmentation, shape &amp; more — <code>segmentation</code> (Felzenszwalb/livewire), <code>shape</code> (shape-context/TPS), <code>ximgproc</code> (EdgeBoxes/SLIC), <code>stitching</code>, <code>hfs</code>, <code>ml</code>, <code>quality</code>, <code>plot</code>, <code>mcc</code>",
    "<b>G-API</b> — <code>gapi</code>, a lazy computation-graph API that compiles a pipeline once and runs it bit-identically to the eager path",
    "<b>CUDA-family (CPU-backed)</b> — <code>cudaarithm</code>/<code>cudaimgproc</code>/<code>cudafilters</code>/<code>cudawarping</code>/… mirror OpenCV's GPU API (same <code>GpuMat</code>/<code>Stream</code>), running on the CPU — parity, not acceleration",
    "Zero dependencies — pure Go standard library, nothing to audit but the toolchain"
  ],
  node_code:
`import cv2

img = cv2.imread("in.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
edges = cv2.Canny(blurred, 50, 150)
cv2.imwrite("edges.png", edges)`,
  go_code:
`import "github.com/malcolmston/opencv"

// The import path is .../opencv, but the package is named cv.
img, _ := cv.ImRead("in.png")
gray := cv.CvtColor(img, cv.ColorRGB2Gray)
blurred := cv.GaussianBlur(gray, 5, 1.4)
edges := cv.Canny(blurred, 50, 150)
cv.ImWrite("edges.png", edges)`,
  integrate:
`<span class="tok-c">// Threshold with an automatically chosen Otsu level, then clean up</span>
<span class="tok-c">// small specks with a morphological opening.</span>
bin, level := cv.Threshold(gray, 0, 255, cv.ThreshBinary|cv.ThreshOtsu)
kernel := cv.GetStructuringElement(cv.MorphRect, 3, 3)
opened := cv.MorphologyEx(bin, kernel, cv.MorphOpen, 1)

<span class="tok-c">// Rotate 30° about the centre via an affine warp.</span>
rot := cv.GetRotationMatrix2D(float64(img.Cols)/2, float64(img.Rows)/2, 30, 1.0)
turned := cv.WarpAffine(img, rot, img.Cols, img.Rows, cv.InterLinear)

<span class="tok-c">// Locate a template and draw a green box + label around the hit.</span>
res := cv.MatchTemplate(gray, templ, cv.TmCcoeffNormed)
_, _, _, _, maxX, maxY := cv.MinMaxLoc(res)
green := cv.NewScalar(0, 255, 0)
cv.Rectangle(img, cv.Point{X: maxX, Y: maxY}, cv.Point{X: maxX + templ.Cols, Y: maxY + templ.Rows}, green, 2)
cv.PutText(img, "match", cv.Point{X: maxX, Y: maxY - 6}, 1, green)`
};
