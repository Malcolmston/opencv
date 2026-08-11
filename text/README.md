# text — from-scratch, standard-library-only port of a useful subset of OpenCV's contrib text

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/text.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/text)

Package text is a from-scratch, standard-library-only port of a useful subset
of OpenCV's contrib text module: classical scene-text detection on top of the
grayscale-and-contours machinery in github.com/malcolmston/opencv (imported
here as cv).

The package does not attempt optical character recognition with a trained
classifier or a neural network. Instead it implements the geometric,
pre-recognition half of a scene-text pipeline: finding candidate character
regions and grouping them into text lines. What you feed to a downstream
recognizer is left to you.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/text"
```

## Usage

This is the package's own `ExampleDetectRegionsMSER`, so it compiles and its output is
asserted on every `go test ./text/`.

```go
img := cv.NewMat(20, 30, 1)
	img.SetTo(220)

	for _, bx := range []int{4, 18} {
		for y := 6; y < 11; y++ {
			for x := bx; x < bx+5; x++ {
				img.Set(y, x, 0, 40)
			}
		}
	}

	boxes := text.DetectRegionsMSER(img, 5, 8, 100, 0.5)
	fmt.Printf("%d regions\n", len(boxes))
```

```
2 regions
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func ComputeNMChannels(img *cv.Mat, mode NMChannelMode) []*cv.Mat` | ComputeNMChannels reproduces the channel decomposition that OpenCV's Neumann–Matas scene-text detector runs its extremal-region search over. |
| `func DetectRegions(img *cv.Mat, p TextDetectorParams) []cv.Rect` | DetectRegions is the convenience entry point of the module: it runs the full pipeline (see `DetectTextLines`) and returns one bounding box per… |
| `func DetectRegionsMSER(img *cv.Mat, delta, minArea, maxArea int, maxVariation float64) []cv.Rect` | DetectRegionsMSER extracts Maximally Stable Extremal Regions from img and returns their bounding boxes. |
| `func DetectRegionsMSERParams(img *cv.Mat, p MSERParams) []cv.Rect` | DetectRegionsMSERParams is the boxes-only counterpart of `MSERRegionsWithParams`, returning just the bounding boxes. |
| `func DetectTextLines(img *cv.Mat, p TextDetectorParams) [][]cv.Rect` | DetectTextLines runs the complete pipeline and returns the surviving character boxes grouped into text lines: extremal-region proposal… |
| `func ERGrouping(boxes []cv.Rect, orientation GroupingOrientation) [][]cv.Rect` | ERGrouping clusters filtered character boxes into text lines. |
| `func ERGroupingBBox(boxes []cv.Rect) [][]cv.Rect` | ERGroupingBBox is the simplified, bounding-box-only grouping OpenCV offers as a fast alternative to the feature-based grouper. |
| `func FontGlyph(ch rune, scale int) (glyph *cv.Mat, ok bool)` | FontGlyph renders one character of the built-in font as a single-channel Mat at the given integer scale (each font pixel becomes a scale×scale… |
| `func GroupTextRegions(regions []cv.Rect) [][]cv.Rect` | GroupTextRegions groups character-like boxes into text lines using the default heuristics (see `DefaultGroupingParams`). |
| `func LineBoundingBox(group []cv.Rect) cv.Rect` | LineBoundingBox returns the smallest axis-aligned rectangle covering every box in a group. |
| `func ProjectionProfile(img *cv.Mat, threshold uint8, vertical bool) []int` | ProjectionProfile returns the ink projection of an image: the number of foreground samples (grayscale value strictly greater than threshold, i.e. |
| `func RenderText(text string, scale, spacing int) *cv.Mat` | RenderText draws a line of built-in-font text into a fresh single-channel Mat: inked glyph pixels are 255 on a 0 background, glyphs are scaled by… |
| `func SegmentChars(img *cv.Mat, threshold uint8) []cv.Rect` | SegmentChars splits a single line of bright-on-dark text into per-character bounding boxes by cutting the vertical ink projection at empty columns,… |
| `func SegmentLines(img *cv.Mat, threshold uint8) []cv.Rect` | SegmentLines splits a multi-line text image into per-line bounding boxes by cutting the horizontal ink projection at empty rows, tightening each line… |
| `func StrokeWidthTransform(img *cv.Mat, p SWTParams) []float64` | StrokeWidthTransform computes the Stroke Width Transform of a single-channel image following Epshtein, Ofek and Wexler (CVPR 2010). |
| `func SupportedChars() string` | SupportedChars returns the characters the built-in font can render and the `OCRTemplate` recognizers can read, in ascending rune order (0-9 then A-Z). |

### Types

| Type | What it is |
| --- | --- |
| `BeamSearchDecoder` | BeamSearchDecoder decodes a per-character score matrix into a word by beam search, optionally constrained to a `Lexicon`. |
| `ERFeatures` | ERFeatures is the geometric and topological descriptor OpenCV's Neumann–Matas Extremal Region classifier is trained on, computed here for a single… |
| `ERFilter` | ERFilter is a lightweight stand-in for OpenCV's trained Extremal Region classifier. |
| `ERFilterNM1` | ERFilterNM1 is the first, high-recall stage of the two-stage Neumann–Matas Extremal Region classifier. |
| `ERFilterNM2` | ERFilterNM2 is the second, high-precision stage of the Neumann–Matas Extremal Region classifier. |
| `GroupingOrientation` | GroupingOrientation selects how `ERGrouping` links character boxes into text lines, mirroring OpenCV's erGrouping ORIENTATION_* modes. |
| `GroupingParams` | GroupingParams controls how `GroupingParams.Group` clusters character boxes into text lines. |
| `Lexicon` | Lexicon is a fixed vocabulary used to constrain a `BeamSearchDecoder`. |
| `MSERParams` | MSERParams collects the tunable parameters of the MSER region detector into one struct, mirroring the arguments of OpenCV's cv::MSER, including the… |
| `MSERPolarity` | MSERPolarity selects the intensity polarity searched by MSER. |
| `NMChannelMode` | NMChannelMode selects the channel decomposition produced by `ComputeNMChannels`, mirroring OpenCV's ERFILTER_NM_* modes. |
| `NearestGlyphClassifier` | NearestGlyphClassifier is a trivial nearest-template recognizer. |
| `OCRTemplate` | OCRTemplate is a nearest-template optical character recognizer over the built-in bitmap font (see `SupportedChars`). |
| `Region` | Region is one detected extremal region: a connected component of an intensity level set that is maximally stable across a range of thresholds. |
| `SWTComponent` | SWTComponent is one stroke-width-connected component found by `TextDetectorSWT.DetectComponents`: its bounding box and the mean and standard… |
| `SWTParams` | SWTParams controls the Stroke Width Transform detector (see `StrokeWidthTransform` and `TextDetectorSWT`). |
| `TextDetectorParams` | TextDetectorParams bundles the parameters of the full classical detection pipeline driven by `DetectRegions` and `DetectTextLines`: MSER region… |
| `TextDetectorSWT` | TextDetectorSWT detects character-like regions with the Stroke Width Transform. |

<details>
<summary><code>BeamSearchDecoder</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewBeamSearchDecoder(alphabet string, beamWidth int) *BeamSearchDecoder` | NewBeamSearchDecoder returns an unconstrained decoder over alphabet with the given beam width. |
| `func (d *BeamSearchDecoder) Decode(scores [][]float64) string` | Decode returns the highest-scoring string for the score matrix. |
| `func (d *BeamSearchDecoder) WithLexicon(lex *Lexicon) *BeamSearchDecoder` | WithLexicon returns a copy of the decoder constrained to lex. |

</details>

<details>
<summary><code>ERFeatures</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func ComputeERFeatures(r Region) ERFeatures` | ComputeERFeatures measures the full `ERFeatures` descriptor of a region. |
| `func (f ERFeatures) StrokeWidthVariation() float64` | StrokeWidthVariation returns StrokeWidthStd/StrokeWidthMean, the coefficient of variation of stroke thickness. |

</details>

<details>
<summary><code>ERFilter</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func DefaultERFilter() ERFilter` | DefaultERFilter returns a filter with permissive character-shaped defaults. |
| `func NewERFilter(minArea, maxArea int) ERFilter` | NewERFilter returns an ERFilter with the given area bounds and the default aspect-ratio and fill-ratio limits. |
| `func (f ERFilter) Filter(regions []Region) []Region` | Filter returns the subset of regions that pass the filter, preserving order. |
| `func (f ERFilter) Keep(r Region) bool` | Keep reports whether a single region passes the filter's shape heuristics. |

</details>

<details>
<summary><code>ERFilterNM1</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func DefaultERFilterNM1() ERFilterNM1` | DefaultERFilterNM1 returns permissive first-stage thresholds tuned to keep printed and hand-drawn characters while discarding degenerate shapes. |
| `func NewERFilterNM1(minArea, maxArea int) ERFilterNM1` | NewERFilterNM1 returns first-stage thresholds with the given area bounds and the default shape limits. |
| `func (f ERFilterNM1) Filter(regions []Region) []Region` | Filter returns the subset of regions that pass stage one, preserving order. |
| `func (f ERFilterNM1) Keep(r Region) bool` | Keep reports whether a region passes stage one, computing its features first. |
| `func (f ERFilterNM1) KeepFeatures(x ERFeatures) bool` | KeepFeatures reports whether a precomputed feature vector passes stage one. |

</details>

<details>
<summary><code>ERFilterNM2</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func DefaultERFilterNM2() ERFilterNM2` | DefaultERFilterNM2 returns second-stage thresholds tuned to admit printed and hand-drawn characters while rejecting concave or variable-stroke… |
| `func NewERFilterNM2(stage1 ERFilterNM1) ERFilterNM2` | NewERFilterNM2 returns second-stage thresholds built on the given first stage and the default holistic limits. |
| `func (f ERFilterNM2) Filter(regions []Region) []Region` | Filter returns the subset of regions that pass both stages, preserving order. |
| `func (f ERFilterNM2) Keep(r Region) bool` | Keep reports whether a region passes both stages, computing its features first. |
| `func (f ERFilterNM2) KeepFeatures(x ERFeatures) bool` | KeepFeatures reports whether a precomputed feature vector passes both stages. |

</details>

<details>
<summary><code>GroupingParams</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func DefaultGroupingParams() GroupingParams` | DefaultGroupingParams returns grouping parameters tuned for evenly spaced, same-height text such as a printed word or line of digits. |
| `func (p GroupingParams) Group(regions []cv.Rect) [][]cv.Rect` | Group clusters regions into text lines according to p. |

</details>

<details>
<summary><code>Lexicon</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewLexicon(words []string) *Lexicon` | NewLexicon builds a lexicon from a word list. |
| `func (l *Lexicon) IsPrefix(s string) bool` | IsPrefix reports whether s is a prefix of (or equal to) some lexicon word. |
| `func (l *Lexicon) IsWord(s string) bool` | IsWord reports whether s is a complete word in the lexicon. |

</details>

<details>
<summary><code>MSERParams</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func DefaultMSERParams() MSERParams` | DefaultMSERParams returns parameters matching OpenCV's cv::MSER defaults scaled for small test images: delta 5, area in [10, 14400], max variation… |

</details>

<details>
<summary><code>NearestGlyphClassifier</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewNearestGlyphClassifier(width, height int, threshold uint8, templates map[string]*cv.Mat) *NearestGlyphClassifier` | NewNearestGlyphClassifier builds a classifier over templates keyed by label. |
| `func (c *NearestGlyphClassifier) Classify(img *cv.Mat) (label string, distance int)` | Classify returns the label of the template closest to img and the Hamming distance (number of differing pixels) to it. |

</details>

<details>
<summary><code>OCRTemplate</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewOCRTemplateAlnum() *OCRTemplate` | NewOCRTemplateAlnum returns a recognizer over the digits 0-9 and the uppercase letters A-Z — the full built-in alphabet. |
| `func NewOCRTemplateDigits() *OCRTemplate` | NewOCRTemplateDigits returns a recognizer restricted to the digits 0-9. |
| `func (o *OCRTemplate) Alphabet() string` | Alphabet returns the characters this recognizer can output, in ascending order. |
| `func (o *OCRTemplate) CharScores(img *cv.Mat) [][]float64` | CharScores classifies each segmented character of one text line and returns, per character, a score for every alphabet symbol. |
| `func (o *OCRTemplate) RecognizeChar(img *cv.Mat) (label string, distance int)` | RecognizeChar classifies a single isolated glyph image, returning its label and the Hamming distance to the matched template. |
| `func (o *OCRTemplate) RecognizeWord(img *cv.Mat) string` | RecognizeWord segments one line of text into characters (`SegmentChars`) and returns the concatenation of their labels, left-to-right. |
| `func (o *OCRTemplate) Run(img *cv.Mat) string` | Run recognizes a full text image: it splits it into lines (`SegmentLines`), reads each line with `OCRTemplate.RecognizeWord`, and joins the lines… |

</details>

<details>
<summary><code>Region</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func MSERRegions(img *cv.Mat, delta, minArea, maxArea int, maxVariation float64) []Region` | MSERRegions is like `DetectRegionsMSER` but returns the full `Region` records, including the pixel set of each region and the threshold and variation… |
| `func MSERRegionsWithParams(img *cv.Mat, p MSERParams) []Region` | MSERRegionsWithParams extracts MSERs under the given `MSERParams`. |

</details>

<details>
<summary><code>SWTParams</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func DefaultSWTParams() SWTParams` | DefaultSWTParams returns parameters tuned for bright strokes on a dark background (DarkOnLight false) with strokes up to 20 px wide. |

</details>

<details>
<summary><code>TextDetectorParams</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func DefaultTextDetectorParams() TextDetectorParams` | DefaultTextDetectorParams returns a pipeline configuration tuned for dark characters on a lighter background (and, via MSER's dual polarity, the… |

</details>

<details>
<summary><code>TextDetectorSWT</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewTextDetectorSWT(p SWTParams) *TextDetectorSWT` | NewTextDetectorSWT returns a detector using the given parameters. |
| `func (d *TextDetectorSWT) Detect(img *cv.Mat) []cv.Rect` | Detect runs `TextDetectorSWT.DetectComponents` and returns just the bounding boxes of the stroke components. |
| `func (d *TextDetectorSWT) DetectComponents(img *cv.Mat) []SWTComponent` | DetectComponents runs the transform and groups pixels of similar stroke width (neighbouring widths within a factor of 3) into components, returning… |

</details>

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/text).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
