# freqdomain — provides frequency-domain image processing on top of the parent cv package's image types

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/freqdomain.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/freqdomain)

Package freqdomain provides frequency-domain image processing on top of the
parent cv package's image types. It implements the classic Fourier toolkit:
forward and inverse 2-D transforms (a fast radix-2 FFT with an exact direct
DFT fallback for non power-of-two sizes), magnitude and phase spectra,
quadrant swapping (fftshift), the standard family of ideal, Butterworth and
Gaussian low-pass, high-pass, band-pass and band-reject filters, notch
filtering, homomorphic filtering, image restoration by inverse and Wiener
deconvolution, and translation estimation by phase correlation.

The central type is `Spectrum`, a complex frequency-domain image stored as two
parallel float64 planes (real and imaginary). It interoperates with the parent
package: real inputs are the cv.FloatMat single-channel float image, and
filter transfer functions are returned as cv.FloatMat so they compose with the
rest of the library and can be visualised with cv.MinMaxLoc and friends.
Convert a cv.Mat (8-bit image) to a working float image with `MatToFloat` and
back with `FloatToMat`.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/freqdomain"
```

## Usage

This is the package's own `ExampleApplyFilter`, so it compiles and its output is
asserted on every `go test ./freqdomain/`.

```go
src := ramp(8, 8)
	allPass := cv.NewFloatMat(8, 8)
	for i := range allPass.Data {
		allPass.Data[i] = 1
	}
	out := freqdomain.ApplyFilter(src, allPass)

	maxErr := 0.0
	for i := range src.Data {
		maxErr = math.Max(maxErr, math.Abs(src.Data[i]-out.Data[i]))
	}
	fmt.Println("all-pass is identity:", maxErr < 1e-9)
```

```
all-pass is identity: true
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func ApplyFilter(f *cv.FloatMat, filter *cv.FloatMat) *cv.FloatMat` | ApplyFilter applies a centred real transfer function to a real image in the frequency domain and returns the real filtered image. |
| `func ApplyWindow(f *cv.FloatMat, window *cv.FloatMat) *cv.FloatMat` | ApplyWindow multiplies a real image element-wise by a window of matching size (such as the Hann window from cv.CreateHanningWindow) and returns a new… |
| `func ButterworthBandPass(rows, cols int, center, width float64, order int) *cv.FloatMat` | ButterworthBandPass returns a centred Butterworth band-pass transfer function, the complement of `ButterworthBandReject`. |
| `func ButterworthBandReject(rows, cols int, center, width float64, order int) *cv.FloatMat` | ButterworthBandReject returns a centred Butterworth band-reject transfer function of centre frequency center and bandwidth width, following Gonzalez:… |
| `func ButterworthHighPass(rows, cols int, cutoff float64, order int) *cv.FloatMat` | ButterworthHighPass returns a centred Butterworth high-pass transfer function H(D) = 1 / (1 + (cutoff/D)^(2·order)), with H(0)=0. |
| `func ButterworthLowPass(rows, cols int, cutoff float64, order int) *cv.FloatMat` | ButterworthLowPass returns a centred Butterworth low-pass transfer function H(D) = 1 / (1 + (D/cutoff)^(2·order)). |
| `func ConvolveFFT(f *cv.FloatMat, kernel *cv.FloatMat) *cv.FloatMat` | ConvolveFFT returns the circular convolution of a real image with a kernel, computed by multiplying their spectra. |
| `func FFT1D(re, im []float64) (outRe, outIm []float64)` | FFT1D returns the forward discrete Fourier transform of a single complex signal given as parallel real and imaginary slices. |
| `func FFTShift(f *cv.FloatMat) *cv.FloatMat` | FFTShift rearranges a spectrum plane so the zero-frequency component moves to the centre of the image (OpenCV / NumPy fftshift). |
| `func FilterImage(m *cv.Mat, filter *cv.FloatMat) *cv.Mat` | FilterImage is a convenience wrapper that filters an 8-bit image with a centred transfer function and returns an 8-bit image. |
| `func FloatToMat(f *cv.FloatMat) *cv.Mat` | FloatToMat converts a single-channel cv.FloatMat to an 8-bit cv.Mat by rounding and clamping each sample to [0,255]. |
| `func FloatToMatScaled(f *cv.FloatMat) *cv.Mat` | FloatToMatScaled linearly rescales a cv.FloatMat so its minimum maps to 0 and its maximum to 255, then rounds to an 8-bit cv.Mat. |
| `func GaussianBandPass(rows, cols int, center, width float64) *cv.FloatMat` | GaussianBandPass returns a centred Gaussian band-pass transfer function, the complement of `GaussianBandReject`. |
| `func GaussianBandReject(rows, cols int, center, width float64) *cv.FloatMat` | GaussianBandReject returns a centred Gaussian band-reject transfer function of centre frequency center and bandwidth width, following Gonzalez: H(D)… |
| `func GaussianHighPass(rows, cols int, cutoff float64) *cv.FloatMat` | GaussianHighPass returns a centred Gaussian high-pass transfer function H(D) = 1 − exp(−D² / (2·cutoff²)), the complement of `GaussianLowPass`. |
| `func GaussianLowPass(rows, cols int, cutoff float64) *cv.FloatMat` | GaussianLowPass returns a centred Gaussian low-pass transfer function H(D) = exp(−D² / (2·cutoff²)). |
| `func GaussianPSF(size int, sigma float64) *cv.FloatMat` | GaussianPSF returns a normalised size×size point-spread function sampled from an isotropic Gaussian of standard deviation sigma, centred in the… |
| `func HomomorphicEmphasis(rows, cols int, gammaLow, gammaHigh, cutoff, c float64) *cv.FloatMat` | HomomorphicEmphasis returns a centred high-frequency-emphasis transfer function of the form |
| `func HomomorphicFilter(m *cv.Mat, gammaLow, gammaHigh, cutoff, c float64) *cv.Mat` | HomomorphicFilter applies homomorphic filtering to an 8-bit image and returns an 8-bit image. |
| `func HomomorphicFilterFloat(f *cv.FloatMat, gammaLow, gammaHigh, cutoff, c float64) *cv.FloatMat` | HomomorphicFilterFloat applies homomorphic filtering to a real image: it takes the natural log of (1 + pixel), filters it in the frequency domain… |
| `func IDFT2D(s *Spectrum) *cv.FloatMat` | IDFT2D returns the real part of the inverse 2-D transform of a spectrum computed by direct summation, normalised by the number of elements. |
| `func IFFT1D(re, im []float64) (outRe, outIm []float64)` | IFFT1D returns the inverse discrete Fourier transform of a single complex signal, normalised by the signal length so that IFFT1D(FFT1D(x)) recovers x. |
| `func IFFT2D(s *Spectrum) *cv.FloatMat` | IFFT2D returns the real part of the inverse 2-D discrete Fourier transform of a spectrum, normalised so that IFFT2D(FFT2D(x)) recovers x. |
| `func IFFTShift(f *cv.FloatMat) *cv.FloatMat` | IFFTShift is the exact inverse of `FFTShift`, moving a centred zero-frequency component back to the top-left corner. |
| `func IdealBandPass(rows, cols int, low, high float64) *cv.FloatMat` | IdealBandPass returns a centred ideal band-pass transfer function: 1 where the distance from the centre lies in [low, high] and 0 elsewhere. |
| `func IdealBandReject(rows, cols int, low, high float64) *cv.FloatMat` | IdealBandReject returns a centred ideal band-reject (band-stop) transfer function, the complement of `IdealBandPass`: 0 where the distance lies in… |
| `func IdealHighPass(rows, cols int, cutoff float64) *cv.FloatMat` | IdealHighPass returns a centred ideal high-pass transfer function, the complement of `IdealLowPass`: 0 within cutoff of the centre and 1 beyond it. |
| `func IdealLowPass(rows, cols int, cutoff float64) *cv.FloatMat` | IdealLowPass returns a centred ideal low-pass transfer function of size rows×cols: 1 where the distance from the centre is at most cutoff and 0… |
| `func InverseFilter(g *cv.FloatMat, psf *cv.FloatMat, threshold float64) *cv.FloatMat` | InverseFilter restores a real degraded image by dividing its spectrum by the optical transfer function of the point-spread function psf. |
| `func IsPowerOfTwo(n int) bool` | IsPowerOfTwo reports whether n is a positive power of two. |
| `func LogMagnitude(s *Spectrum) *cv.FloatMat` | LogMagnitude returns log(1+\|F\|) of a spectrum, the standard compression used to visualise a Fourier magnitude whose dynamic range spans many orders… |
| `func MagnitudeSpectrum(f *cv.FloatMat) *cv.Mat` | MagnitudeSpectrum returns a display-ready magnitude spectrum of a real image: the forward transform, log(1+\|F\|) compression, an fftshift to centre… |
| `func MatToFloat(m *cv.Mat) *cv.FloatMat` | MatToFloat converts an 8-bit cv.Mat to a single-channel cv.FloatMat of samples in [0,255]. |
| `func MotionBlurPSF(size int, angle float64) *cv.FloatMat` | MotionBlurPSF returns a normalised size×size point-spread function modelling linear motion blur of the given angle in degrees, measured… |
| `func NextPowerOfTwo(n int) int` | NextPowerOfTwo returns the smallest power of two that is greater than or equal to n. |
| `func NotchPass(rows, cols int, notches []Notch) *cv.FloatMat` | NotchPass returns a centred Butterworth notch-pass transfer function, the complement of `NotchReject`, which keeps only the frequencies near the… |
| `func NotchReject(rows, cols int, notches []Notch) *cv.FloatMat` | NotchReject returns a centred Butterworth notch-reject transfer function of size rows×cols that suppresses each listed `Notch` and its conjugate… |
| `func PhaseCorrelation(a, b *cv.FloatMat) (dx, dy int, response float64)` | PhaseCorrelation estimates the circular translational shift between two real images of equal size using the phase-correlation method: it forms the… |
| `func PhaseSpectrum(f *cv.FloatMat) *cv.Mat` | PhaseSpectrum returns a display-ready phase spectrum of a real image: the forward transform, an fftshift to centre the DC term and a linear rescale… |
| `func RegisterTranslation(a, b *cv.FloatMat) (dx, dy int)` | RegisterTranslation returns only the integer (dx, dy) shift that best aligns image a onto image b, discarding the response value. |
| `func WienerDeconvolution(g *cv.FloatMat, psf *cv.FloatMat, nsr float64) *cv.FloatMat` | WienerDeconvolution restores a real degraded image using the Wiener filter |

### Types

| Type | What it is |
| --- | --- |
| `Notch` | Notch describes a single conjugate notch centred at frequency offset (U, V) relative to the spectrum centre. |
| `Spectrum` | Spectrum is a complex-valued frequency-domain image stored as two parallel planes of float64 samples in row-major order. |

<details>
<summary><code>Spectrum</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func ApplyFilterSpectrum(s *Spectrum, filter *cv.FloatMat) *Spectrum` | ApplyFilterSpectrum applies a centred real transfer function to a spectrum and returns the filtered spectrum. |
| `func CrossPowerSpectrum(a, b *Spectrum) *Spectrum` | CrossPowerSpectrum returns the normalised cross-power spectrum of two spectra, |
| `func DFT2D(f *cv.FloatMat) *Spectrum` | DFT2D returns the forward 2-D transform of a real image computed by direct summation. |
| `func FFT2D(f *cv.FloatMat) *Spectrum` | FFT2D returns the forward 2-D discrete Fourier transform of a real image as a complex `Spectrum`. |
| `func FFT2DComplex(s *Spectrum) *Spectrum` | FFT2DComplex returns the forward 2-D discrete Fourier transform of a complex image. |
| `func FFTShiftSpectrum(s *Spectrum) *Spectrum` | FFTShiftSpectrum applies `FFTShift` to both planes of a complex spectrum, returning a new spectrum with the DC term centred. |
| `func IFFT2DComplex(s *Spectrum) *Spectrum` | IFFT2DComplex returns the inverse 2-D discrete Fourier transform of a complex spectrum, normalised by the number of elements so that… |
| `func IFFTShiftSpectrum(s *Spectrum) *Spectrum` | IFFTShiftSpectrum is the exact inverse of `FFTShiftSpectrum`, moving a centred DC term back to the corner of a complex spectrum. |
| `func NewSpectrum(rows, cols int) *Spectrum` | NewSpectrum allocates a zero-filled Spectrum with the given dimensions. |
| `func PSFToOTF(psf *cv.FloatMat, rows, cols int) *Spectrum` | PSFToOTF converts a spatial point-spread function into an optical transfer function (its FFT) of the requested size. |
| `func SpectrumFromComplex(re, im *cv.FloatMat) *Spectrum` | SpectrumFromComplex builds a Spectrum from separate real and imaginary planes of matching size. |
| `func SpectrumFromFloat(f *cv.FloatMat) *Spectrum` | SpectrumFromFloat builds a Spectrum from a real image, setting the imaginary plane to zero. |
| `func (s *Spectrum) Add(other *Spectrum) *Spectrum` | Add returns the element-wise complex sum of s and other. |
| `func (s *Spectrum) At(y, x int) (re, im float64)` | At returns the complex sample at row y, column x as its real and imaginary parts. |
| `func (s *Spectrum) Clone() *Spectrum` | Clone returns a deep copy of the spectrum. |
| `func (s *Spectrum) Conjugate() *Spectrum` | Conjugate returns a new spectrum with every sample complex-conjugated (the imaginary part negated). |
| `func (s *Spectrum) ImagPlane() *cv.FloatMat` | ImagPlane returns the imaginary part of the spectrum as a fresh cv.FloatMat. |
| `func (s *Spectrum) Magnitude() *cv.FloatMat` | Magnitude returns the per-element magnitude sqrt(re²+im²) of the spectrum as a cv.FloatMat. |
| `func (s *Spectrum) Mul(other *Spectrum) *Spectrum` | Mul returns the element-wise complex product of s and other. |
| `func (s *Spectrum) MulReal(mask *cv.FloatMat) *Spectrum` | MulReal returns a new spectrum with every complex sample multiplied by the corresponding real gain from mask, a real transfer function of matching… |
| `func (s *Spectrum) Phase() *cv.FloatMat` | Phase returns the per-element phase angle atan2(im, re) of the spectrum in radians (range (-π, π]) as a cv.FloatMat. |
| `func (s *Spectrum) PowerSpectrum() *cv.FloatMat` | PowerSpectrum returns the per-element power \|F\|² = re²+im² of the spectrum as a cv.FloatMat. |
| `func (s *Spectrum) RealPlane() *cv.FloatMat` | RealPlane returns the real part of the spectrum as a fresh cv.FloatMat. |
| `func (s *Spectrum) Scale(factor float64) *Spectrum` | Scale returns a new spectrum with every sample multiplied by the real scalar factor. |
| `func (s *Spectrum) Set(y, x int, re, im float64)` | Set stores the complex sample re+im·i at row y, column x. |
| `func (s *Spectrum) Size() (rows, cols int)` | Size returns the spectrum dimensions as (rows, cols). |

</details>

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/freqdomain).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
