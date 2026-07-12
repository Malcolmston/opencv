package cudacodec

import "image"

// Codec identifies a video codec, mirroring the constants of OpenCV's
// cudacodec::Codec enumeration. In upstream OpenCV these select an NVDEC/NVENC
// hardware path; here they express intent only. The concrete on-disk container
// is chosen from the output file extension (see [CreateVideoWriter]), and the
// requested Codec is preserved through [FormatInfo.Codec] for round-tripping and
// diagnostics. Compressed values such as [CodecH264] and [CodecHEVC] are honoured
// by substitution with the Motion-JPEG AVI container.
type Codec int

// Codec values matching cudacodec::Codec. The ordered block mirrors OpenCV's
// enumerator order so numeric values ported from OpenCV keep their meaning.
const (
	CodecMPEG1     Codec = iota // MPEG-1
	CodecMPEG2                  // MPEG-2
	CodecMPEG4                  // MPEG-4 part 2
	CodecVC1                    // SMPTE VC-1
	CodecH264                   // H.264 / AVC
	CodecJPEG                   // Motion-JPEG (the native substitution codec)
	CodecH264SVC                // H.264 SVC
	CodecH264MVC                // H.264 MVC
	CodecHEVC                   // H.265 / HEVC
	CodecVP8                    // VP8
	CodecVP9                    // VP9
	CodecAV1                    // AV1
	CodecNumCodecs              // sentinel: number of compressed codecs

	// CodecUncompressedRGBA marks a lossless, uncompressed RGBA stream. It is the
	// substitution codec reported for the APNG and GIF containers, which carry
	// full frames rather than a compressed bitstream.
	CodecUncompressedRGBA
)

// codecNames backs [Codec.String]; entries beyond the ordered block are handled
// specially.
var codecNames = [...]string{
	CodecMPEG1:   "MPEG1",
	CodecMPEG2:   "MPEG2",
	CodecMPEG4:   "MPEG4",
	CodecVC1:     "VC1",
	CodecH264:    "H264",
	CodecJPEG:    "JPEG",
	CodecH264SVC: "H264_SVC",
	CodecH264MVC: "H264_MVC",
	CodecHEVC:    "HEVC",
	CodecVP8:     "VP8",
	CodecVP9:     "VP9",
	CodecAV1:     "AV1",
}

// String returns the codec's OpenCV-style name.
func (c Codec) String() string {
	switch c {
	case CodecNumCodecs:
		return "NumCodecs"
	case CodecUncompressedRGBA:
		return "Uncompressed_RGBA"
	}
	if int(c) >= 0 && int(c) < len(codecNames) && codecNames[c] != "" {
		return codecNames[c]
	}
	return "Unknown"
}

// ColorFormat identifies the pixel layout of the frames exchanged with a reader
// or writer, mirroring cudacodec::ColorFormat. The cv toolkit stores frames as
// interleaved 8-bit RGB or grayscale [cv.Mat] values, so a ColorFormat is
// advisory metadata recorded in [FormatInfo]; no channel reordering is performed
// on the pixel data.
type ColorFormat int

// ColorFormat values matching cudacodec::ColorFormat.
const (
	ColorFormatUndefined ColorFormat = iota // no preference
	ColorFormatBGRA                         // 4-channel BGRA
	ColorFormatBGR                          // 3-channel BGR
	ColorFormatGray                         // single-channel grayscale
	ColorFormatNV12                         // NV12 (planar YUV 4:2:0)
	ColorFormatRGB                          // 3-channel RGB
	ColorFormatRGBA                         // 4-channel RGBA
)

// colorFormatNames backs [ColorFormat.String].
var colorFormatNames = [...]string{
	ColorFormatUndefined: "UNDEFINED",
	ColorFormatBGRA:      "BGRA",
	ColorFormatBGR:       "BGR",
	ColorFormatGray:      "GRAY",
	ColorFormatNV12:      "NV_NV12",
	ColorFormatRGB:       "RGB",
	ColorFormatRGBA:      "RGBA",
}

// String returns the color format's OpenCV-style name.
func (f ColorFormat) String() string {
	if int(f) >= 0 && int(f) < len(colorFormatNames) {
		return colorFormatNames[f]
	}
	return "Unknown"
}

// SurfaceFormat identifies the encoder input surface layout, mirroring
// cudacodec::SurfaceFormat (the SF_* enumerators). As with [ColorFormat], it is
// advisory metadata in this port and does not repack pixel data.
type SurfaceFormat int

// SurfaceFormat values matching cudacodec::SurfaceFormat.
const (
	SurfaceFormatUYVY SurfaceFormat = iota // packed YUV 4:2:2 (UYVY)
	SurfaceFormatYUY2                      // packed YUV 4:2:2 (YUY2)
	SurfaceFormatYV12                      // planar YUV 4:2:0 (YV12)
	SurfaceFormatNV12                      // planar YUV 4:2:0 (NV12)
	SurfaceFormatIYUV                      // planar YUV 4:2:0 (IYUV)
	SurfaceFormatBGR                       // packed 8-bit BGR
	SurfaceFormatGray                      // single-channel grayscale
)

// surfaceFormatNames backs [SurfaceFormat.String].
var surfaceFormatNames = [...]string{
	SurfaceFormatUYVY: "SF_UYVY",
	SurfaceFormatYUY2: "SF_YUY2",
	SurfaceFormatYV12: "SF_YV12",
	SurfaceFormatNV12: "SF_NV12",
	SurfaceFormatIYUV: "SF_IYUV",
	SurfaceFormatBGR:  "SF_BGR",
	SurfaceFormatGray: "SF_GRAY",
}

// String returns the surface format's OpenCV-style name.
func (s SurfaceFormat) String() string {
	if int(s) >= 0 && int(s) < len(surfaceFormatNames) {
		return surfaceFormatNames[s]
	}
	return "Unknown"
}

// ChromaFormat identifies chroma subsampling, mirroring
// cudacodec::ChromaFormat. It appears in [FormatInfo]; the substitution codecs
// used here decode to full RGB, so it is reported as advisory metadata.
type ChromaFormat int

// ChromaFormat values matching cudacodec::ChromaFormat.
const (
	ChromaMonochrome ChromaFormat = iota // no chroma
	ChromaYUV420                         // 4:2:0
	ChromaYUV422                         // 4:2:2
	ChromaYUV444                         // 4:4:4
)

// chromaNames backs [ChromaFormat.String].
var chromaNames = [...]string{
	ChromaMonochrome: "Monochrome",
	ChromaYUV420:     "YUV420",
	ChromaYUV422:     "YUV422",
	ChromaYUV444:     "YUV444",
}

// String returns the chroma format's OpenCV-style name.
func (c ChromaFormat) String() string {
	if int(c) >= 0 && int(c) < len(chromaNames) {
		return chromaNames[c]
	}
	return "Unknown"
}

// FormatInfo describes the format of a decoded stream, mirroring
// cudacodec::FormatInfo. A reader fills it in from the container it opened; the
// fields relevant to a pure-Go, CPU-decoded stream are populated and the rest
// carry sensible defaults. Obtain one from [VideoReader.Format].
type FormatInfo struct {
	// Codec is the (possibly substituted) codec the stream is treated as.
	Codec Codec
	// Chroma is the advisory chroma subsampling of the source.
	Chroma ChromaFormat
	// NBitDepthMinus8 is the sample bit depth minus 8; always 0 (8-bit) here.
	NBitDepthMinus8 int
	// Width is the coded frame width in pixels.
	Width int
	// Height is the coded frame height in pixels.
	Height int
	// DisplayArea is the region of each frame intended for display. It spans the
	// full frame in this port.
	DisplayArea image.Rectangle
	// FPS is the nominal frame rate in frames per second, or 0 if unspecified.
	FPS float64
	// NumFrames is the total number of frames in the stream.
	NumFrames int
	// Valid reports whether the reader successfully determined the format.
	Valid bool
}

// EncoderParams mirrors cudacodec::EncoderParams, the tuning bundle passed to a
// GPU encoder. NVENC exposes dozens of rate-control knobs; this port keeps a
// small, meaningful subset and applies what the standard-library codecs can
// honour (notably the frame rate). The remaining fields are recorded and echoed
// back so ported code that sets them still compiles and round-trips.
type EncoderParams struct {
	// FPS is the target playback rate in frames per second.
	FPS float64
	// SurfaceFormat is the advisory encoder input surface layout.
	SurfaceFormat SurfaceFormat
	// Quality is a 0..100 hint (higher is better); advisory for the MJPEG path.
	Quality int
	// GOPLength is the requested key-frame interval; advisory (every frame in the
	// MJPEG/APNG/GIF substitutions is effectively a key frame).
	GOPLength int
}

// NewEncoderParams returns EncoderParams populated with reasonable defaults: 25
// fps, NV12 surface format, quality 95 and a one-second GOP.
func NewEncoderParams() EncoderParams {
	return EncoderParams{
		FPS:           25,
		SurfaceFormat: SurfaceFormatNV12,
		Quality:       95,
		GOPLength:     25,
	}
}
