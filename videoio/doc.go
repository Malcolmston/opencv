// Package videoio provides a small, standard-library-only "video" layer for the
// cv image-processing toolkit ([github.com/malcolmston/opencv]). It reads and
// writes sequences of frames using codecs that ship with the Go standard
// library, so it builds and runs anywhere the Go toolchain does — with no cgo
// and no third-party dependencies.
//
// A clip is treated as a short sequence of frames, each carrying a delay
// measured in hundredths of a second (centiseconds). Several containers are
// supported, all built on standard-library codecs:
//
//   - Animated GIF (image/gif) — palette-limited, via [ReadGIF]/[WriteGIF], the
//     configurable [PalettedGIFWriter] (choice of [PaletteWebSafe],
//     [PalettePlan9] or an [AdaptivePalette]) and per-frame-delay [WriteGIFDelays].
//   - Animated PNG / APNG (image/png plus a hand-rolled chunk layer) — a true
//     lossless format, via [ReadAPNG]/[WriteAPNG], [APNGWriter] and
//     [WriteAPNGDelays]. A real acTL/fcTL/fdAT chunk stream is assembled and
//     parsed, and plain PNGs decode as a single frame.
//   - Motion-JPEG AVI (image/jpeg inside a real RIFF/AVI container) — via
//     [ReadMJPEGAVI]/[WriteMJPEGAVI] and [AVIWriter]. The writer emits a genuine
//     RIFF structure (avih, strh, strf BITMAPINFOHEADER, movi, idx1) of
//     concatenated JPEG frames; the reader walks it back.
//   - Numbered image sequences (frame0001.png, …) — via [ImageSequenceWriter],
//     [ReadImageSequence]/[WriteImageSequence] and [OpenImageSequence].
//
// [WriteVideoFromMats] and [ReadVideoToMats] dispatch to the right container by
// file extension.
//
// # Captures, properties and seeking
//
// Every reader materialises its frames into a [VideoCapture], which offers the
// OpenCV-style property model: [VideoCapture.Get] and [VideoCapture.Set] read
// and write CAP_PROP_* values (FPS, frame count, width, height, position),
// [VideoCapture.Grab]/[VideoCapture.Retrieve] mirror OpenCV's two-step read, and
// [VideoCapture.SetPosFrames] seeks. [OpenGIF], [OpenAPNG], [OpenAVI] and
// [OpenImageSequence] all return one and satisfy the [FrameGrabber] interface.
// [ResampleFrames] and [ResampleCapture] retime a clip from one frame rate to
// another by nearest-frame sampling.
//
// # Relationship to OpenCV
//
// The names mirror OpenCV's video I/O API so the code reads familiarly:
// [VideoCapture] pulls decoded frames one at a time, and [VideoWriter]
// accumulates frames and encodes them on release. Unlike OpenCV, which relies
// on ffmpeg and a pile of native codecs to open MP4/AVI/etc., this package is
// deliberately limited to what the pure-Go standard library can do. Real
// compressed-video codecs (H.264, VP9, …) require cgo bindings to ffmpeg or a
// similar library and remain out of scope.
//
// # Frames as Mats
//
// Every frame crosses the boundary as a [cv.Mat]. Decoding composites the GIF's
// (possibly partial, possibly disposed) sub-images onto a full-size canvas and
// converts the result with [cv.FromImage], yielding three-channel RGB Mats.
// Encoding converts each Mat back to an image with [cv.Mat.ToImage] and
// quantizes it to a 256-colour palette. Because frame sizes in a clip must
// agree, the writer adopts the bounds of the first frame it receives and places
// every later frame at the origin of that canvas, clipping any overflow.
//
// # Quantization and fidelity
//
// GIF is a paletted format: each frame may use at most 256 distinct colours.
// This package quantizes every frame to the fixed 216-colour web-safe palette
// ([palette.WebSafe]) by nearest-colour mapping, without dithering, so the
// output is fully deterministic — the same frames always produce byte-identical
// files. The cost is colour fidelity: each channel of every pixel may shift by
// up to roughly 26 levels toward the nearest palette entry. Callers that need
// exact colour should keep the original Mats; the GIF round-trip is lossy by
// construction.
//
// # Typical use
//
// Read an animated GIF and iterate its frames:
//
//	cap, err := videoio.OpenGIF("clip.gif")
//	if err != nil {
//		log.Fatal(err)
//	}
//	defer cap.Close()
//	for {
//		frame, ok := cap.Read()
//		if !ok {
//			break
//		}
//		_ = frame // a *cv.Mat
//	}
//
// Write a sequence of Mats to an animated GIF at 10 frames per second
// (delay = 10 centiseconds per frame):
//
//	w, err := videoio.NewGIFWriter("out.gif", 10)
//	if err != nil {
//		log.Fatal(err)
//	}
//	for _, m := range frames {
//		if err := w.Write(m); err != nil {
//			log.Fatal(err)
//		}
//	}
//	if err := w.Release(); err != nil {
//		log.Fatal(err)
//	}
//
// The convenience functions [ReadGIF] and [WriteGIF] wrap the whole read or
// write in a single call when streaming is not required.
package videoio
