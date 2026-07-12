// Package videoio provides a small, standard-library-only "video" layer for the
// cv image-processing toolkit ([github.com/malcolmston/opencv]). It reads and
// writes sequences of frames using codecs that ship with the Go standard
// library, so it builds and runs anywhere the Go toolchain does — with no cgo
// and no third-party dependencies.
//
// The only container format the standard library can both decode and encode as
// a multi-frame animation is the GIF, so that is the format this package speaks.
// A GIF is treated as a short, palette-limited video clip: each stored image is
// a frame and each frame carries a delay measured in hundredths of a second
// (centiseconds), exactly as the GIF format records it.
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
