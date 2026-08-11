# videoio — provides a small, standard-library-only "video" layer for the cv image-processing

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/videoio.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/videoio)

Package videoio provides a small, standard-library-only "video" layer for the
cv image-processing toolkit (`github.com/malcolmston/opencv`). It reads and
writes sequences of frames using codecs that ship with the Go standard
library, so it builds and runs anywhere the Go toolchain does — with no cgo
and no third-party dependencies.

- Animated GIF (image/gif) — palette-limited, via `ReadGIF`/`WriteGIF`, the
configurable `PalettedGIFWriter` (choice of `PaletteWebSafe`, `PalettePlan9`
or an `AdaptivePalette`) and per-frame-delay `WriteGIFDelays`. - Animated PNG
/ APNG (image/png plus a hand-rolled chunk layer) — a true lossless format,
via `ReadAPNG`/`WriteAPNG`, `APNGWriter` and `WriteAPNGDelays`. A real
acTL/fcTL/fdAT chunk stream is assembled and parsed, and plain PNGs decode as
a single frame. - Motion-JPEG AVI (image/jpeg inside a real RIFF/AVI
container) — via `ReadMJPEGAVI`/`WriteMJPEGAVI` and `AVIWriter`. The writer
emits a genuine RIFF structure (avih, strh, strf BITMAPINFOHEADER, movi, idx1)
of concatenated JPEG frames; the reader walks it back. - Numbered image
sequences (frame0001.png, …) — via `ImageSequenceWriter`,
`ReadImageSequence`/`WriteImageSequence` and `OpenImageSequence`.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/videoio"
```

## Usage

This is the package's own `ExampleResampleFrames`, so it compiles and its output is
asserted on every `go test ./videoio/`.

```go
frames := []*cv.Mat{cv.NewMat(2, 2, 3), cv.NewMat(2, 2, 3)}
	delays := []int{50, 50}
	out, outDelays := videoio.ResampleFrames(frames, delays, 10)
	fmt.Printf("%d frames, %d cs each\n", len(out), outDelays[0])
```

```
10 frames, 10 cs each
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func AdaptivePalette(frames []*cv.Mat, maxColors int) color.Palette` | AdaptivePalette builds a palette of at most maxColors entries tailored to the given frames, using popularity quantization: every pixel is binned into… |
| `func FourCC(a, b, c, d byte) uint32` | FourCC packs four ASCII characters into a 32-bit code the way video containers identify codecs (for example 'M','J','P','G'). |
| `func FourCCString(code uint32) string` | FourCCString unpacks a code produced by `FourCC` back into its four characters. |
| `func ReadAPNG(path string) ([]*cv.Mat, []int, error)` | ReadAPNG decodes the APNG (or plain PNG) at path and returns every frame as a three-channel RGB Mat together with the matching per-frame delays in… |
| `func ReadGIF(path string) ([]*cv.Mat, []int, error)` | ReadGIF decodes the GIF at path and returns every frame as a Mat together with the matching per-frame delays in centiseconds. |
| `func ReadImageSequence(dir, pattern string, start int) ([]*cv.Mat, error)` | ReadImageSequence loads a numbered image sequence from dir. |
| `func ReadMJPEGAVI(path string) ([]*cv.Mat, float64, error)` | ReadMJPEGAVI parses the MJPEG AVI at path, JPEG-decodes every frame in its movi list into a three-channel RGB Mat, and returns the frames together… |
| `func ReadVideoToMats(path string) ([]*cv.Mat, float64, error)` | ReadVideoToMats decodes the clip at path, choosing the container from the file extension (".gif", ".png"/".apng" or ".avi"), and returns every frame… |
| `func ResampleFrames(frames []*cv.Mat, delaysCentis []int, targetFPS float64) ([]*cv.Mat, []int)` | ResampleFrames resamples a clip to a constant target frame rate. |
| `func WriteAPNG(path string, frames []*cv.Mat, delayCentis int) error` | WriteAPNG encodes frames into an APNG at path, giving every frame the same delay of delayCentis centiseconds. |
| `func WriteAPNGDelays(path string, frames []*cv.Mat, delays []int) error` | WriteAPNGDelays encodes frames into an APNG at path with an independent delay (in centiseconds) for each frame, enabling variable-rate playback. |
| `func WriteGIF(path string, frames []*cv.Mat, delayCentis int) error` | WriteGIF encodes frames into an animated GIF at path, giving every frame the same delay of delayCentis centiseconds. |
| `func WriteGIFDelays(path string, frames []*cv.Mat, delays []int, pal color.Palette, loopCount int) error` | WriteGIFDelays encodes frames to an animated GIF at path with an independent per-frame delay (centiseconds) and a chosen palette and loop count. |
| `func WriteImageSequence(dir, pattern string, frames []*cv.Mat, start int) ([]string, error)` | WriteImageSequence writes every frame to dir using pattern, numbering them from start, and returns the paths written. |
| `func WriteMJPEGAVI(path string, frames []*cv.Mat, fps float64) error` | WriteMJPEGAVI encodes frames into a Motion-JPEG AVI at path, played back at fps frames per second. |
| `func WriteVideoFromMats(path string, frames []*cv.Mat, fps float64) error` | WriteVideoFromMats encodes frames to path, choosing the container from the file extension: ".gif" writes an animated GIF, ".png"/".apng" writes an… |

### Types

| Type | What it is |
| --- | --- |
| `APNGWriter` | APNGWriter accumulates frames and encodes them as a single Animated PNG when released. |
| `AVIWriter` | AVIWriter accumulates frames and writes them as a minimal but standards-shaped Motion-JPEG AVI file when released. |
| `FrameGrabber` | FrameGrabber is the read side of the video I/O API, modelled on OpenCV's cv::VideoCapture. |
| `ImageSequenceWriter` | ImageSequenceWriter saves frames as individually numbered image files in a directory — the classic "frame0001.png, frame0002.png, …" layout… |
| `PalettedGIFWriter` | PalettedGIFWriter encodes frames to an animated GIF with caller-controlled quantization: any `color.Palette` may be supplied (a fixed one such as… |
| `PropID` | PropID identifies a capture or writer property. |
| `VideoCapture` | VideoCapture reads the frames of an animated GIF as a sequence of Mats. |
| `VideoWriter` | VideoWriter accumulates frames and encodes them as a single animated GIF when released. |

<details>
<summary><code>APNGWriter</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewAPNGWriter(path string, delayCentis int) (*APNGWriter, error)` | NewAPNGWriter creates a writer that encodes every frame passed to `APNGWriter.Write` into an APNG file at path when `APNGWriter.Release` runs. |
| `func (w *APNGWriter) Release() error` | Release encodes every written frame into the destination APNG and finalizes the writer. |
| `func (w *APNGWriter) SetLoopCount(n int)` | SetLoopCount limits how many times viewers replay the animation; 0 (the default) means loop forever. |
| `func (w *APNGWriter) Write(frame *cv.Mat) error` | Write appends frame to the animation, using the writer's default per-frame delay. |
| `func (w *APNGWriter) WriteFrame(frame *cv.Mat, delayCentis int) error` | WriteFrame appends frame with an explicit display duration of delayCentis centiseconds, overriding the writer's default for this frame only. |

</details>

<details>
<summary><code>AVIWriter</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewAVIWriter(path string, fps float64) (*AVIWriter, error)` | NewAVIWriter creates a writer that emits an MJPEG AVI at path when `AVIWriter.Release` runs, played back at fps frames per second. |
| `func (w *AVIWriter) Release() error` | Release assembles and writes the AVI file, then finalizes the writer. |
| `func (w *AVIWriter) Write(frame *cv.Mat) error` | Write JPEG-encodes frame and appends it to the movi stream. |

</details>

<details>
<summary><code>ImageSequenceWriter</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewImageSequenceWriter(dir, pattern string, index int) (*ImageSequenceWriter, error)` | NewImageSequenceWriter creates a writer that places frames in dir. |
| `func (w *ImageSequenceWriter) Count() int` | Count returns the number of frames written so far. |
| `func (w *ImageSequenceWriter) Files() []string` | Files returns the paths written so far, in order. |
| `func (w *ImageSequenceWriter) Write(frame *cv.Mat) (string, error)` | Write encodes frame to the next numbered file and returns its path. |

</details>

<details>
<summary><code>PalettedGIFWriter</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewPalettedGIFWriter(path string, pal color.Palette, loopCount int) (*PalettedGIFWriter, error)` | NewPalettedGIFWriter creates a GIF writer that maps every frame onto pal. |
| `func (w *PalettedGIFWriter) Release() error` | Release encodes every frame to the destination GIF and finalizes the writer. |
| `func (w *PalettedGIFWriter) WriteFrame(frame *cv.Mat, delayCentis int) error` | WriteFrame quantizes frame to the writer's palette and appends it with a display duration of delayCentis centiseconds, so each frame may play for a… |

</details>

<details>
<summary><code>VideoCapture</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func OpenAPNG(path string) (*VideoCapture, error)` | OpenAPNG decodes the APNG at path and returns a `VideoCapture` over its frames, so an animated PNG can be streamed with the same grab/retrieve API as… |
| `func OpenAVI(path string) (*VideoCapture, error)` | OpenAVI parses the MJPEG AVI at path and returns a `VideoCapture` over its frames, with per-frame delays derived from the file's frame rate so that… |
| `func OpenGIF(path string) (*VideoCapture, error)` | OpenGIF opens the animated (or single-frame) GIF at path and decodes every frame into memory. |
| `func OpenImageSequence(dir, pattern string, start, delayCentis int) (*VideoCapture, error)` | OpenImageSequence reads a numbered sequence with `ReadImageSequence` and returns a `VideoCapture` over the frames, so an on-disk frame directory can… |
| `func ResampleCapture(c *VideoCapture, targetFPS float64) *VideoCapture` | ResampleCapture resamples a capture's frames to targetFPS and returns a fresh `VideoCapture` positioned at the start, so a clip opened at one rate… |
| `func (c *VideoCapture) Close() error` | Close releases the decoded frames and resets the capture. |
| `func (c *VideoCapture) Delays() []int` | Delays returns the per-frame display durations in centiseconds (hundredths of a second), one entry per frame and in the same order as… |
| `func (c *VideoCapture) FrameCount() int` | FrameCount returns the total number of decoded frames. |
| `func (c *VideoCapture) Frames() []*cv.Mat` | Frames returns all decoded frames in order. |
| `func (c *VideoCapture) Get(prop PropID) float64` | Get returns the value of property prop, or 0 if the property is unknown or the capture is empty. |
| `func (c *VideoCapture) Grab() bool` | Grab advances the capture to the next frame without copying it out, returning true while frames remain. |
| `func (c *VideoCapture) PosFrames() int` | PosFrames returns the zero-based index of the frame that the next `VideoCapture.Read` or `VideoCapture.Grab` will return. |
| `func (c *VideoCapture) Read() (*cv.Mat, bool)` | Read returns the next frame and true, advancing the internal position. |
| `func (c *VideoCapture) Retrieve() (*cv.Mat, bool)` | Retrieve returns the frame selected by the most recent successful `VideoCapture.Grab`. |
| `func (c *VideoCapture) Set(prop PropID, value float64) bool` | Set writes property prop and reports whether the property is settable on a capture. |
| `func (c *VideoCapture) SetPosFrames(n int) int` | SetPosFrames seeks so that the next read returns frame n. |

</details>

<details>
<summary><code>VideoWriter</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewGIFWriter(path string, delayCentis int) (*VideoWriter, error)` | NewGIFWriter creates a writer that will encode all frames given to `VideoWriter.Write` into an animated GIF at path when `VideoWriter.Release` is… |
| `func (w *VideoWriter) Get(prop PropID) float64` | Get returns the value of property prop for the writer, or 0 if it is unknown. |
| `func (w *VideoWriter) Release() error` | Release encodes every written frame to the destination GIF and finalizes the writer. |
| `func (w *VideoWriter) Set(prop PropID, value float64) bool` | Set writes property prop and reports whether it was applied. |
| `func (w *VideoWriter) Write(frame *cv.Mat) error` | Write quantizes frame and appends it to the animation. |

</details>

### Variables

`PalettePlan9`, `PaletteWebSafe`

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/videoio).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
