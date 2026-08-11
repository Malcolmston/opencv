# datamatrix — self-contained, dependency-free ECC200 Data Matrix codec for the stdlib-only OpenCV port

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/datamatrix.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/datamatrix)

Package datamatrix is a self-contained, dependency-free ECC200 Data Matrix
codec for the stdlib-only OpenCV port. It encodes ASCII strings into square
Data Matrix symbol bitmaps and detects and decodes those bitmaps back into the
original text, correcting a bounded number of errors along the way.

- ASCII encodation: printable and control ASCII bytes map to codeword value+1,
and adjacent digit pairs are packed into a single codeword (value+130),
exactly as in ISO/IEC 16022. - Symbol selection: the smallest supported square
symbol whose data capacity fits is chosen automatically. - Padding: a single
end-of-message codeword (129) followed by the 253-state randomised pad fills
any remaining data capacity. - Reed-Solomon: error-correction codewords are
generated over GF(256) using the ECC200 primitive polynomial 0x12D, generator
element 2 and first root exponent 1, reproducing the standard generator
polynomials. - Placement: codeword bits are laid into the mapping matrix with
the ECC200 "utah" placement algorithm (ISO/IEC 16022 Annex F), including the
corner special cases and the bottom-right fixed pattern. - Finder/timing: a
solid "L" (left column and bottom row) plus an alternating clock track (top
row and right column) frame the symbol.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/datamatrix"
```

## Usage

This is the package's own `ExampleDecodeAll`, so it compiles and its output is
asserted on every `go test ./datamatrix/`.

```go
opts := datamatrix.Options{ModulePixels: 3, QuietZoneModules: 2, Channels: 1}
	a, _ := datamatrix.EncodeText("ALPHA", datamatrix.EncodeOptions{Scheme: datamatrix.SchemeASCII, Render: opts})
	b, _ := datamatrix.EncodeText("BETA", datamatrix.EncodeOptions{Scheme: datamatrix.SchemeASCII, Render: opts})
	canvas := cv.NewMat(a.Rows+12, a.Cols+b.Cols+24, 1)
	canvas.SetTo(255)
	a.CopyTo(canvas, 6, 6)
	b.CopyTo(canvas, 6, a.Cols+18)
	results, _ := datamatrix.DecodeAll(canvas)
	fmt.Printf("found %d symbols\n", len(results))
```

```
found 2 symbols
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func DecodeMatrix(modules [][]bool) (string, error)` | DecodeMatrix decodes a fully-sampled square module grid, where modules`r``c` is true for a dark module. |
| `func DetectAndDecode(m *cv.Mat) (string, error)` | DetectAndDecode locates a single Data Matrix symbol in the image, samples its module grid, and decodes it. |
| `func Encode(text string) (*cv.Mat, error)` | Encode encodes an ASCII string into an ECC200 Data Matrix bitmap using `DefaultOptions`. |
| `func EncodeText(text string, opts EncodeOptions) (*cv.Mat, error)` | EncodeText encodes text into an ECC200 Data Matrix bitmap using the supplied options, supporting every encodation scheme, square and rectangular… |
| `func EncodeWithOptions(text string, opts Options) (*cv.Mat, error)` | EncodeWithOptions is like `Encode` but lets the caller control the module pixel size, quiet-zone width and channel count. |

### Types

| Type | What it is |
| --- | --- |
| `DecodedResult` | DecodedResult is the outcome of decoding one symbol, including any metadata. |
| `EncodeOptions` | EncodeOptions controls the extended `EncodeText` encoder. |
| `EncodedSymbol` | EncodedSymbol is a fully laid-out ECC200 symbol of any supported size. |
| `Options` | Options controls how a symbol is rendered into a bitmap. |
| `Scheme` | Scheme selects the encodation scheme used for the message content. |
| `SizePreference` | SizePreference selects which symbol shapes `EncodeText` may choose from. |
| `StructuredAppend` | StructuredAppend describes a symbol's place in a structured-append group, in which a long message is split across up to 16 Data Matrix symbols that a… |
| `Symbol` | Symbol is a fully-laid-out square Data Matrix symbol. |

<details>
<summary><code>DecodedResult</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func DecodeAll(m *cv.Mat) ([]*DecodedResult, error)` | DecodeAll locates and decodes every Data Matrix symbol in the image, returning one result per symbol. |
| `func DecodeGrid(modules [][]bool) (*DecodedResult, error)` | DecodeGrid decodes a fully-sampled module grid of any supported square or rectangular size, returning the content and metadata. |
| `func DecodeText(m *cv.Mat) (*DecodedResult, error)` | DecodeText locates a single Data Matrix symbol of any supported size in the image and decodes it, returning the content and metadata. |

</details>

<details>
<summary><code>EncodedSymbol</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func EncodeTextSymbol(text string, opts EncodeOptions) (*EncodedSymbol, error)` | EncodeTextSymbol is the lower-level counterpart to `EncodeText`: it returns the laid-out `EncodedSymbol` (module grid) for callers that want the… |
| `func (e *EncodedSymbol) Render(opts Options) *cv.Mat` | Render rasterises the symbol into a Mat according to opts, matching the conventions of `EncodeWithOptions`: dark modules are black on a white… |

</details>

<details>
<summary><code>Options</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func DefaultOptions() Options` | DefaultOptions returns the rendering options used by `Encode`: 8 pixels per module, a 2-module quiet zone and a single-channel (grayscale) bitmap. |

</details>

<details>
<summary><code>Symbol</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func EncodeSymbol(text string) (*Symbol, error)` | EncodeSymbol encodes text into a laid-out `Symbol` (the module grid) and also reports the chosen symbol specification's side length. |

</details>

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/datamatrix).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
