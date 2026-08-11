# cudacodec — CPU-backed, API-compatible mirror of OpenCV's cudacodec module for the cv image toolkit

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/cudacodec.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/cudacodec)

Package cudacodec is a CPU-backed, API-compatible mirror of OpenCV's cudacodec
module for the cv image toolkit (`github.com/malcolmston/opencv`). It
reproduces the shape of OpenCV's hardware video-codec API — `VideoReader`,
`VideoWriter`, the `Codec`, `ColorFormat` and `SurfaceFormat` enumerations,
`EncoderParams` and `FormatInfo` — but contains no cgo, no CUDA and no GPU
code whatsoever. It builds and runs anywhere the Go toolchain does.

In upstream OpenCV, cudacodec wraps NVIDIA's NVDEC/NVENC hardware engines to
decode and encode compressed video (H.264, HEVC, VP9, AV1, …) directly on
the GPU, exchanging frames as device-resident cv::cuda::GpuMat objects. None
of that is reachable from the pure-Go standard library: the hardware codecs
require proprietary drivers and native SDKs behind cgo.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/cudacodec"
```

## Usage

This is the package's own `Example`, so it compiles and its output is
asserted on every `go test ./cudacodec/`.

```go
path := filepath.Join(os.TempDir(), "cudacodec_example_clip.avi")

	w, err := cudacodec.CreateVideoWriter(path, image.Pt(24, 16),
		cudacodec.CodecH264, 25, cudacodec.ColorFormatBGR)
	if err != nil {
		fmt.Println("writer:", err)
		return
	}
	for i := 0; i < 4; i++ {
		if err := w.Write(cudacodec.NewGpuMatFromMat(gradient(i, 24, 16))); err != nil {
			fmt.Println("write:", err)
			return
		}
	}
	if err := w.Release(); err != nil {
		fmt.Println("release:", err)
		return
	}

	r, err := cudacodec.CreateVideoReader(path)
	if err != nil {
		fmt.Println("reader:", err)
		return
	}
	defer r.Release()

	count := 0
	dst := cudacodec.NewGpuMat()
	for r.NextFrame(dst) {
		count++
	}
	fmt.Printf("decoded %d frames, requested codec substituted as %v\n",
		count, r.Format().Codec)
```

```
decoded 4 frames, requested codec substituted as JPEG
```

## Exported surface

### Types

| Type | What it is |
| --- | --- |
| `ChromaFormat` | ChromaFormat identifies chroma subsampling, mirroring cudacodec::ChromaFormat. |
| `Codec` | Codec identifies a video codec, mirroring the constants of OpenCV's cudacodec::Codec enumeration. |
| `ColorFormat` | ColorFormat identifies the pixel layout of the frames exchanged with a reader or writer, mirroring cudacodec::ColorFormat. |
| `EncoderParams` | EncoderParams mirrors cudacodec::EncoderParams, the tuning bundle passed to a GPU encoder. |
| `FormatInfo` | FormatInfo describes the format of a decoded stream, mirroring cudacodec::FormatInfo. |
| `GpuMat` | GpuMat mirrors OpenCV's cv::cuda::GpuMat, the device-resident matrix that the real cudacodec API exchanges with NVDEC/NVENC. |
| `Stream` | Stream mirrors OpenCV's cv::cuda::Stream, the handle that queues asynchronous GPU work. |
| `SurfaceFormat` | SurfaceFormat identifies the encoder input surface layout, mirroring cudacodec::SurfaceFormat (the SF_* enumerators). |
| `VideoReader` | VideoReader mirrors cudacodec::VideoReader, the GPU-accelerated decoder of the real module. |
| `VideoWriter` | VideoWriter mirrors cudacodec::VideoWriter, the GPU-accelerated encoder of the real module. |

<details>
<summary><code>ChromaFormat</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (c ChromaFormat) String() string` | String returns the chroma format's OpenCV-style name. |

</details>

<details>
<summary><code>Codec</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (c Codec) String() string` | String returns the codec's OpenCV-style name. |

</details>

<details>
<summary><code>ColorFormat</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (f ColorFormat) String() string` | String returns the color format's OpenCV-style name. |

</details>

<details>
<summary><code>EncoderParams</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewEncoderParams() EncoderParams` | NewEncoderParams returns EncoderParams populated with reasonable defaults: 25 fps, NV12 surface format, quality 95 and a one-second GOP. |

</details>

<details>
<summary><code>GpuMat</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewGpuMat() *GpuMat` | NewGpuMat returns an empty GpuMat. |
| `func NewGpuMatFromMat(m *cv.Mat) *GpuMat` | NewGpuMatFromMat returns a GpuMat backed by a deep copy of m, so later mutations of m do not affect the GpuMat. |
| `func (g *GpuMat) Channels() int` | Channels returns the number of samples per pixel, or 0 when empty. |
| `func (g *GpuMat) Clone() *GpuMat` | Clone returns an independent deep copy of the GpuMat. |
| `func (g *GpuMat) Cols() int` | Cols returns the frame width in pixels, or 0 when empty. |
| `func (g *GpuMat) Download(_ ...*Stream) *cv.Mat` | Download copies the GpuMat back to a host `cv.Mat`, mirroring cv::cuda::GpuMat::download. |
| `func (g *GpuMat) Empty() bool` | Empty reports whether the GpuMat currently holds no frame. |
| `func (g *GpuMat) Mat() *cv.Mat` | Mat returns the GpuMat's underlying host matrix without copying, or nil when empty. |
| `func (g *GpuMat) Release()` | Release frees the GpuMat's frame, mirroring cv::cuda::GpuMat::release. |
| `func (g *GpuMat) Rows() int` | Rows returns the frame height in pixels, or 0 when empty. |
| `func (g *GpuMat) Size() image.Point` | Size returns the frame size as an image.Point in OpenCV's (width, height) order: X is the number of columns and Y the number of rows. |
| `func (g *GpuMat) Upload(m *cv.Mat, _ ...*Stream)` | Upload copies a host `cv.Mat` into the GpuMat, mirroring cv::cuda::GpuMat::upload. |

</details>

<details>
<summary><code>Stream</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewStream() *Stream` | NewStream returns a no-op `Stream`. |
| `func (s *Stream) QueryIfComplete() bool` | QueryIfComplete reports whether all queued work has finished, mirroring cv::cuda::Stream::queryIfComplete. |
| `func (s *Stream) WaitForCompletion()` | WaitForCompletion blocks until all queued work on the stream finishes, mirroring cv::cuda::Stream::waitForCompletion. |

</details>

<details>
<summary><code>SurfaceFormat</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (s SurfaceFormat) String() string` | String returns the surface format's OpenCV-style name. |

</details>

<details>
<summary><code>VideoReader</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func CreateVideoReader(filename string) (*VideoReader, error)` | CreateVideoReader opens the clip at filename and returns a `VideoReader`, mirroring cudacodec::createVideoReader. |
| `func (r *VideoReader) Format() FormatInfo` | Format returns the `FormatInfo` describing the open stream, mirroring cudacodec::VideoReader::format. |
| `func (r *VideoReader) FrameCount() int` | FrameCount returns the total number of frames in the stream. |
| `func (r *VideoReader) Get(prop videoio.PropID) float64` | Get returns the value of capture property prop, mirroring cudacodec::VideoReader::get. |
| `func (r *VideoReader) Grab(_ ...*Stream) bool` | Grab advances to the next frame without decoding it into a destination, mirroring cudacodec::VideoReader::grab. |
| `func (r *VideoReader) NextFrame(dst *GpuMat, _ ...*Stream) bool` | NextFrame decodes the next frame into dst and reports whether a frame was produced, mirroring cudacodec::VideoReader::nextFrame. |
| `func (r *VideoReader) Release() error` | Release frees the reader and its decoded frames, mirroring the destruction of a cudacodec::VideoReader. |
| `func (r *VideoReader) Retrieve(dst *GpuMat, _ ...*Stream) bool` | Retrieve returns the frame selected by the most recent successful `VideoReader.Grab`, mirroring cudacodec::VideoReader::retrieve. |

</details>

<details>
<summary><code>VideoWriter</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func CreateVideoWriter(filename string, frameSize image.Point, codec Codec, fps float64, colorFormat ColorFormat) (*VideoWriter, error)` | CreateVideoWriter opens filename for writing and returns a `VideoWriter`, mirroring cudacodec::createVideoWriter. |
| `func CreateVideoWriterWithParams(filename string, frameSize image.Point, codec Codec, colorFormat ColorFormat, params EncoderParams) (*VideoWriter, error)` | CreateVideoWriterWithParams is like `CreateVideoWriter` but takes an `EncoderParams` bundle, mirroring the params overload of… |
| `func (w *VideoWriter) Codec() Codec` | Codec returns the codec the writer was created with, echoing the intent recorded at construction. |
| `func (w *VideoWriter) ColorFormat() ColorFormat` | ColorFormat returns the advisory color format the writer was created with. |
| `func (w *VideoWriter) Get(prop videoio.PropID) float64` | Get returns writer metadata by videoio CAP_PROP_* identifier: CAP_PROP_FPS, CAP_PROP_FRAME_COUNT (frames written so far), and CAP_PROP_FRAME_WIDTH /… |
| `func (w *VideoWriter) Release() error` | Release finalizes the container and flushes it to disk, mirroring cudacodec::VideoWriter::release. |
| `func (w *VideoWriter) Write(frame *GpuMat) error` | Write encodes one frame, mirroring cudacodec::VideoWriter::write. |

</details>

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/cudacodec).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
