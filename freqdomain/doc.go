// Package freqdomain provides frequency-domain image processing on top of the
// parent cv package's image types. It implements the classic Fourier toolkit:
// forward and inverse 2-D transforms (a fast radix-2 FFT with an exact direct
// DFT fallback for non power-of-two sizes), magnitude and phase spectra,
// quadrant swapping (fftshift), the standard family of ideal, Butterworth and
// Gaussian low-pass, high-pass, band-pass and band-reject filters, notch
// filtering, homomorphic filtering, image restoration by inverse and Wiener
// deconvolution, and translation estimation by phase correlation.
//
// # Types
//
// The central type is [Spectrum], a complex frequency-domain image stored as
// two parallel float64 planes (real and imaginary). It interoperates with the
// parent package: real inputs are the cv.FloatMat single-channel float image,
// and filter transfer functions are returned as cv.FloatMat so they compose
// with the rest of the library and can be visualised with cv.MinMaxLoc and
// friends. Convert a cv.Mat (8-bit image) to a working float image with
// [MatToFloat] and back with [FloatToMat].
//
// # Conventions
//
// Frequency filters are defined on a centred spectrum, with the zero-frequency
// (DC) term at coordinate (rows/2, cols/2). The distance D(u,v) used by every
// filter is the Euclidean distance from that centre. The high-level entry
// points [ApplyFilter] and [FilterImage] handle the FFT, the centring shift and
// the inverse transform internally, so a caller only needs a transfer function
// such as the one returned by [GaussianLowPass] or [ButterworthHighPass].
//
// All routines are deterministic, CPU-only and depend only on the Go standard
// library and the parent cv package.
package freqdomain
