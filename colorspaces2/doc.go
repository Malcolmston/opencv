// Package colorspaces2 provides extended colour-science utilities that build on
// the root cv package's [cv.Mat] image type.
//
// The package is organised around two layers:
//
//   - Scalar colour values and conversions. The small value types [RGB], [HSV],
//     [HSL], [XYZ], [Lab], [Luv], [YCbCr], [YUV] and [CMYK] hold one colour in
//     floating-point form, and the RGBToX / XToRGB functions convert between
//     them with the standard CIE / ITU formulae. RGB values are sRGB,
//     gamma-encoded and normalised to [0,1]; the exact ranges of the other
//     spaces are documented on each type.
//
//   - Image operations on [cv.Mat]. Gamma correction, gray-world and white-patch
//     white balance, colour-temperature adjustment, Bradford chromatic
//     adaptation and colour quantisation (uniform, median-cut and k-means) all
//     operate on three-channel RGB Mats and return new Mats, leaving the input
//     unmodified.
//
// Everything is pure Go, deterministic and CPU-only. A three-channel Mat is
// interpreted as RGB (matching [cv.Mat.ToImage]); functions that require three
// channels panic otherwise, consistent with the root package's conventions.
package colorspaces2
