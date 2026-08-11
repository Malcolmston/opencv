# dnn — small, standard-library-only feed-forward neural-network inference engine

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/dnn.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/dnn)

Package dnn is a small, standard-library-only feed-forward neural-network
inference engine, a from-scratch port of a useful subset of OpenCV's dnn
module. It runs pre-trained convolutional and fully-connected networks that
you assemble and weight by hand in Go — there is no model-file parsing, no
autodiff and no training.

The package builds on the root cv package (github.com/malcolmston/opencv) only
at its image boundary: `BlobFromImage` turns a `cv.Mat` into a network input
and `BlobToImage` turns an output back into a Mat. Everything in between is
plain Go and the standard library (math). It uses no cgo and no third-party
code, and it does not import the other cv/* subpackages.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/dnn"
```

## Usage

This is the package's own `ExampleArgMax`, so it compiles and its output is
asserted on every `go test ./dnn/`.

```go
in := dnn.NewTensorFrom([]int{2, 3}, []float64{1, 5, 2, 9, 0, 3})
	out := (&dnn.ArgMax{Axis: -1, KeepDims: false}).Forward([]*dnn.Tensor{in})[0]
	fmt.Println(out.Data)
```

```
[1 0]
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func BlobToImage(t *Tensor, index int, scale float64, mean []float64, swapRB bool) *cv.Mat` | BlobToImage is the inverse of `BlobFromImage`: it reconstructs a `cv.Mat` from one image of an NCHW blob, undoing the normalization with |
| `func NMSBoxes(boxes []Box, scores []float64, scoreThreshold, nmsThreshold float64) []int` | NMSBoxes performs greedy non-maximum suppression, the standard filter applied to a detector's raw boxes. |

### Types

| Type | What it is |
| --- | --- |
| `Add` | Add sums two or more tensors of identical shape elementwise, returning one tensor. |
| `ArgMax` | ArgMax reduces a tensor along one axis to the index of its maximum element, returning those indices as float64 samples. |
| `AvgPool2D` | AvgPool2D downsamples an NCHW input by averaging each pooling window. |
| `BatchNorm` | BatchNorm applies inference-time batch normalization per channel: |
| `Box` | Box is an axis-aligned rectangle given by its top-left corner and size, the [x, y, width, height] convention used by OpenCV detection heads. |
| `Classification` | Classification pairs a class index with its score, as produced by `ClassifyTopK`. |
| `Concat` | Concat joins several tensors along one axis. |
| `Conv2D` | Conv2D is a 2-D convolution (technically a cross-correlation, matching every deep-learning framework and OpenCV's dnn) over a batched NCHW input. |
| `ConvTranspose2D` | ConvTranspose2D is a 2-D transposed convolution (also called deconvolution or fractionally-strided convolution) over a batched NCHW input. |
| `Dense` | Dense is an alias for `FullyConnected`, provided to match the naming used by higher-level frameworks. |
| `Dropout` | Dropout is an inference-time no-op. |
| `ELU` | ELU applies the exponential-linear unit elementwise: |
| `Eltwise` | Eltwise combines two or more tensors of identical shape with an elementwise reduction — sum, product or maximum — and returns a single tensor of… |
| `EltwiseOp` | EltwiseOp selects the reduction an `Eltwise` layer applies across its inputs. |
| `Flatten` | Flatten collapses every axis after the batch axis into one, mapping a tensor of shape [N, d1, d2, …] to [N, d1*d2*…]. |
| `FullyConnected` | FullyConnected is a dense (fully-connected / inner-product) layer. |
| `GlobalAvgPool` | GlobalAvgPool collapses the spatial extent of an NCHW tensor by averaging each channel over its entire height and width, mapping [N, C, H, W] to [N,… |
| `LRN` | LRN performs local response normalization over an NCHW tensor. |
| `Layer` | Layer is a single stage of a neural network. |
| `LeakyReLU` | LeakyReLU applies x for x >= 0 and Alpha*x for x < 0, elementwise. |
| `MaxPool2D` | MaxPool2D downsamples an NCHW input by taking the maximum over each pooling window. |
| `Mish` | Mish applies the self-regularized non-monotonic activation |
| `Net` | Net is a feed-forward network: an ordered list of named `Layer`s evaluated in sequence. |
| `PReLU` | PReLU applies the parametric rectified-linear activation |
| `Padding` | Padding enlarges a tensor by adding a constant-valued border. |
| `Permute` | Permute reorders the axes of a tensor, moving data so the result is contiguous in the new order. |
| `ReLU` | ReLU applies the rectified-linear activation max(0, x) elementwise, leaving the shape unchanged. |
| `Reshape` | Reshape reinterprets its input under a new shape without moving data. |
| `Sequential` | Sequential is a fluent builder for a linear `Net`. |
| `Sigmoid` | Sigmoid applies the logistic sigmoid 1/(1+exp(-x)) elementwise. |
| `Slice` | Slice extracts a strided sub-range along a single axis of its input, like the Python expression x[..., Start:End:Step, ...] applied to Axis. |
| `Softmax` | Softmax applies a numerically stable softmax along a single axis, turning that axis into a probability distribution that sums to 1. |
| `Swish` | Swish applies the gated activation y = x * sigmoid(Beta*x) elementwise. |
| `Tanh` | Tanh applies the hyperbolic tangent elementwise. |
| `Tensor` | Tensor is a dense, row-major n-dimensional array of float64 samples together with its integer shape. |
| `Transpose` | Transpose swaps two axes of a tensor, moving data. |
| `Upsample` | Upsample enlarges the spatial dimensions of an NCHW tensor by integer factors ScaleH and ScaleW, mapping [N, C, H, W] to [N, C, H*ScaleH, W*ScaleW]. |
| `UpsampleMode` | UpsampleMode selects the interpolation used by `Upsample`. |

<details>
<summary><code>Add</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (a *Add) Forward(inputs []*Tensor) []*Tensor` | Forward returns the elementwise sum of all input tensors. |

</details>

<details>
<summary><code>ArgMax</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewArgMax(axis int) *ArgMax` | NewArgMax builds an ArgMax over the given axis that keeps the reduced axis. |
| `func (a *ArgMax) Forward(inputs []*Tensor) []*Tensor` | Forward computes the argmax of the single input tensor along Axis. |

</details>

<details>
<summary><code>AvgPool2D</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewAvgPool2D(kernel, stride int) *AvgPool2D` | NewAvgPool2D builds an average-pooling layer with a square kernel and given stride, no padding. |
| `func NewAvgPool2DPad(kernelH, kernelW, strideH, strideW, padH, padW int) *AvgPool2D` | NewAvgPool2DPad builds an average-pooling layer with independent kernel, stride and padding extents per axis, matching the full parameter set of the… |
| `func (a *AvgPool2D) Forward(inputs []*Tensor) []*Tensor` | Forward average-pools the single NCHW input tensor. |
| `func (a *AvgPool2D) OutputShape(inShape []int) []int` | OutputShape returns the NCHW shape this layer produces for the given input shape, without allocating or running anything. |

</details>

<details>
<summary><code>BatchNorm</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewBatchNorm(gamma, beta, mean, variance *Tensor, eps float64) *BatchNorm` | NewBatchNorm builds a batch-normalization layer from per-channel mean and variance, optional scale (gamma) and shift (beta), and an epsilon. |
| `func (b *BatchNorm) Forward(inputs []*Tensor) []*Tensor` | Forward normalizes the single input tensor along its channel axis (axis 1). |

</details>

<details>
<summary><code>Box</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (b Box) IoU(other Box) float64` | IoU returns the intersection-over-union overlap of b and other in [0,1]. |

</details>

<details>
<summary><code>Classification</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func ClassifyTopK(scores *Tensor, k int) []Classification` | ClassifyTopK returns the k highest-scoring classes from a classifier output, sorted by descending score. |

</details>

<details>
<summary><code>Concat</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewConcat(axis int) *Concat` | NewConcat returns a Concat over the given axis. |
| `func (c *Concat) Forward(inputs []*Tensor) []*Tensor` | Forward concatenates all input tensors along Axis and returns one tensor. |

</details>

<details>
<summary><code>Conv2D</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewConv2D(weights, bias *Tensor, stride, pad, dilation int) *Conv2D` | NewConv2D builds a convolution with symmetric stride, padding and dilation. |
| `func (c *Conv2D) Forward(inputs []*Tensor) []*Tensor` | Forward convolves the single NCHW input tensor and returns one NCHW output. |

</details>

<details>
<summary><code>ConvTranspose2D</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewConvTranspose2D(weights, bias *Tensor, stride, pad, dilation int) *ConvTranspose2D` | NewConvTranspose2D builds a transposed convolution with symmetric stride, padding and dilation and no output padding. |
| `func (c *ConvTranspose2D) Forward(inputs []*Tensor) []*Tensor` | Forward transposed-convolves the single NCHW input and returns one NCHW output. |

</details>

<details>
<summary><code>Dense</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewDense(weights, bias *Tensor) *Dense` | NewDense is an alias for `NewFullyConnected`. |

</details>

<details>
<summary><code>Dropout</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewDropout(rate float64) *Dropout` | NewDropout returns a Dropout recording the given training-time rate. |
| `func (d *Dropout) Forward(inputs []*Tensor) []*Tensor` | Forward returns a copy of the single input tensor unchanged. |

</details>

<details>
<summary><code>ELU</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewELU(alpha float64) *ELU` | NewELU returns an ELU with the given alpha. |
| `func (e *ELU) Forward(inputs []*Tensor) []*Tensor` | Forward applies the ELU to the single input tensor. |

</details>

<details>
<summary><code>Eltwise</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewEltwise(op EltwiseOp) *Eltwise` | NewEltwise builds an Eltwise with the given operation and no coefficients. |
| `func NewEltwiseSum(coeffs []float64) *Eltwise` | NewEltwiseSum builds a weighted elementwise sum. |
| `func (e *Eltwise) Forward(inputs []*Tensor) []*Tensor` | Forward reduces all input tensors elementwise according to Op. |

</details>

<details>
<summary><code>Flatten</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (f *Flatten) Forward(inputs []*Tensor) []*Tensor` | Forward flattens the single input tensor to rank 2. |

</details>

<details>
<summary><code>FullyConnected</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewFullyConnected(weights, bias *Tensor) *FullyConnected` | NewFullyConnected builds a dense layer. |
| `func (f *FullyConnected) Forward(inputs []*Tensor) []*Tensor` | Forward computes y = x·Wᵀ + b for the single input tensor of shape [N, inFeatures]. |

</details>

<details>
<summary><code>GlobalAvgPool</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (g *GlobalAvgPool) Forward(inputs []*Tensor) []*Tensor` | Forward global-average-pools the single NCHW input tensor. |

</details>

<details>
<summary><code>LRN</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewLRN(size int, alpha, beta, k float64) *LRN` | NewLRN builds an across-channel LRN with the given window size and coefficients and NormBySize enabled (the common configuration). |
| `func (l *LRN) Forward(inputs []*Tensor) []*Tensor` | Forward normalizes the single NCHW input tensor, preserving its shape. |

</details>

<details>
<summary><code>LeakyReLU</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewLeakyReLU(alpha float64) *LeakyReLU` | NewLeakyReLU returns a leaky ReLU with the given negative slope. |
| `func (l *LeakyReLU) Forward(inputs []*Tensor) []*Tensor` | Forward applies the leaky ReLU to the single input tensor. |

</details>

<details>
<summary><code>MaxPool2D</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewMaxPool2D(kernel, stride int) *MaxPool2D` | NewMaxPool2D builds a max-pooling layer with a square kernel and given stride, no padding. |
| `func NewMaxPool2DPad(kernelH, kernelW, strideH, strideW, padH, padW int) *MaxPool2D` | NewMaxPool2DPad builds a max-pooling layer with independent kernel, stride and padding extents per axis, matching the full parameter set of the ONNX… |
| `func (m *MaxPool2D) Forward(inputs []*Tensor) []*Tensor` | Forward max-pools the single NCHW input tensor. |
| `func (m *MaxPool2D) OutputShape(inShape []int) []int` | OutputShape returns the NCHW shape this layer produces for the given input shape, without allocating or running anything. |

</details>

<details>
<summary><code>Mish</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewMish() *Mish` | NewMish returns a Mish activation layer. |
| `func (m *Mish) Forward(inputs []*Tensor) []*Tensor` | Forward applies Mish to the single input tensor. |

</details>

<details>
<summary><code>Net</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewNet() *Net` | NewNet returns an empty network. |
| `func (n *Net) Add(layer Layer) *Net` | Add appends layer with an automatically generated name ("layer0", "layer1", …) and returns the receiver so calls can be chained. |
| `func (n *Net) AddNamed(name string, layer Layer) *Net` | AddNamed appends layer under the given name and returns the receiver. |
| `func (n *Net) Forward(input *Tensor) *Tensor` | Forward runs input through every layer in order and returns the final tensor. |
| `func (n *Net) ForwardMulti(inputs []*Tensor) []*Tensor` | ForwardMulti runs inputs through the network, passing the full output slice of each layer as the input slice of the next. |
| `func (n *Net) LayerByName(name string) (Layer, bool)` | LayerByName returns the layer registered under name and whether it was found. |
| `func (n *Net) LayerNames() []string` | LayerNames returns the layer names in evaluation order. |
| `func (n *Net) Len() int` | Len returns the number of layers in the network. |

</details>

<details>
<summary><code>PReLU</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewPReLU(slope *Tensor) *PReLU` | NewPReLU builds a PReLU from a rank-1 slope tensor. |
| `func (p *PReLU) Forward(inputs []*Tensor) []*Tensor` | Forward applies the PReLU to the single input tensor, preserving its shape. |

</details>

<details>
<summary><code>Padding</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewPadding(begin, end []int, value float64) *Padding` | NewPadding builds a constant Padding with the given per-axis begin and end amounts and fill value. |
| `func NewSpatialPadding(top, bottom, left, right int, value float64) *Padding` | NewSpatialPadding builds a Padding for a rank-4 NCHW tensor that pads only the height and width axes by the given amounts on each side. |
| `func (p *Padding) Forward(inputs []*Tensor) []*Tensor` | Forward pads the single input tensor. |

</details>

<details>
<summary><code>Permute</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewPermute(order ...int) *Permute` | NewPermute builds a Permute with the given axis order. |
| `func (p *Permute) Forward(inputs []*Tensor) []*Tensor` | Forward permutes the axes of the single input tensor. |

</details>

<details>
<summary><code>ReLU</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewReLU() *ReLU` | NewReLU returns a ReLU activation layer. |
| `func (r *ReLU) Forward(inputs []*Tensor) []*Tensor` | Forward applies ReLU to the single input tensor. |

</details>

<details>
<summary><code>Reshape</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewReshape(shape ...int) *Reshape` | NewReshape builds a Reshape to the given target shape. |
| `func (r *Reshape) Forward(inputs []*Tensor) []*Tensor` | Forward reshapes the single input tensor. |

</details>

<details>
<summary><code>Sequential</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewSequential() *Sequential` | NewSequential returns an empty Sequential builder. |
| `func (s *Sequential) Add(layer Layer) *Sequential` | Add appends an arbitrary layer with an auto-generated name. |
| `func (s *Sequential) ArgMax(axis int) *Sequential` | ArgMax appends an argmax-reduction layer over the given axis (keeping dims). |
| `func (s *Sequential) AvgPool2D(p *AvgPool2D) *Sequential` | AvgPool2D appends an average-pooling layer. |
| `func (s *Sequential) BatchNorm(b *BatchNorm) *Sequential` | BatchNorm appends a batch-normalization layer. |
| `func (s *Sequential) Build() *Net` | Build finalizes and returns the assembled network. |
| `func (s *Sequential) Conv2D(c *Conv2D) *Sequential` | Conv2D appends a 2-D convolution layer. |
| `func (s *Sequential) ConvTranspose2D(c *ConvTranspose2D) *Sequential` | ConvTranspose2D appends a transposed-convolution (deconvolution) layer. |
| `func (s *Sequential) Dense(d *FullyConnected) *Sequential` | Dense appends a fully-connected layer. |
| `func (s *Sequential) Dropout(rate float64) *Sequential` | Dropout appends an inference-time no-op dropout recording the given rate. |
| `func (s *Sequential) ELU(alpha float64) *Sequential` | ELU appends an exponential-linear-unit activation. |
| `func (s *Sequential) Flatten() *Sequential` | Flatten appends a layer that flattens each batch element to a vector. |
| `func (s *Sequential) GlobalAvgPool() *Sequential` | GlobalAvgPool appends a global average-pooling layer. |
| `func (s *Sequential) LRN(l *LRN) *Sequential` | LRN appends a local-response-normalization layer. |
| `func (s *Sequential) LeakyReLU(alpha float64) *Sequential` | LeakyReLU appends a leaky rectified-linear activation with the given slope for negative inputs. |
| `func (s *Sequential) MaxPool2D(p *MaxPool2D) *Sequential` | MaxPool2D appends a max-pooling layer. |
| `func (s *Sequential) Mish() *Sequential` | Mish appends a Mish activation. |
| `func (s *Sequential) Named(name string, layer Layer) *Sequential` | Named appends an arbitrary layer under an explicit name. |
| `func (s *Sequential) PReLU(slope *Tensor) *Sequential` | PReLU appends a parametric-ReLU activation with the given slope tensor. |
| `func (s *Sequential) Padding(p *Padding) *Sequential` | Padding appends a constant-padding layer. |
| `func (s *Sequential) Permute(order ...int) *Sequential` | Permute appends an axis-permutation layer. |
| `func (s *Sequential) ReLU() *Sequential` | ReLU appends a rectified-linear activation. |
| `func (s *Sequential) Reshape(shape ...int) *Sequential` | Reshape appends a layer that reshapes the input to the given shape. |
| `func (s *Sequential) SiLU() *Sequential` | SiLU appends a SiLU (Swish with beta 1) activation. |
| `func (s *Sequential) Sigmoid() *Sequential` | Sigmoid appends a logistic-sigmoid activation. |
| `func (s *Sequential) Slice(sl *Slice) *Sequential` | Slice appends a single-axis slice layer. |
| `func (s *Sequential) Softmax() *Sequential` | Softmax appends a softmax over the last axis. |
| `func (s *Sequential) Swish(beta float64) *Sequential` | Swish appends a Swish activation with the given beta. |
| `func (s *Sequential) Tanh() *Sequential` | Tanh appends a hyperbolic-tangent activation. |
| `func (s *Sequential) Transpose(axis1, axis2 int) *Sequential` | Transpose appends a two-axis transpose layer. |
| `func (s *Sequential) Upsample(u *Upsample) *Sequential` | Upsample appends a spatial upsampling layer. |

</details>

<details>
<summary><code>Sigmoid</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewSigmoid() *Sigmoid` | NewSigmoid returns a logistic sigmoid activation layer. |
| `func (s *Sigmoid) Forward(inputs []*Tensor) []*Tensor` | Forward applies the sigmoid to the single input tensor. |

</details>

<details>
<summary><code>Slice</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewSlice(axis, start, end int) *Slice` | NewSlice builds a Slice over one axis with Step 1. |
| `func (s *Slice) Forward(inputs []*Tensor) []*Tensor` | Forward extracts the configured sub-range from the single input tensor. |

</details>

<details>
<summary><code>Softmax</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewSoftmax() *Softmax` | NewSoftmax returns a Softmax over the last axis. |
| `func NewSoftmaxAxis(axis int) *Softmax` | NewSoftmaxAxis returns a Softmax over the given axis. |
| `func (s *Softmax) Forward(inputs []*Tensor) []*Tensor` | Forward applies the softmax to the single input tensor, preserving its shape. |

</details>

<details>
<summary><code>Swish</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewSiLU() *Swish` | NewSiLU returns the SiLU activation, a `Swish` with Beta = 1: y = x*sigmoid(x). |
| `func NewSwish(beta float64) *Swish` | NewSwish returns a Swish with the given beta. |
| `func (s *Swish) Forward(inputs []*Tensor) []*Tensor` | Forward applies the Swish/SiLU activation to the single input tensor. |

</details>

<details>
<summary><code>Tanh</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewTanh() *Tanh` | NewTanh returns a hyperbolic-tangent activation layer. |
| `func (t *Tanh) Forward(inputs []*Tensor) []*Tensor` | Forward applies tanh to the single input tensor. |

</details>

<details>
<summary><code>Tensor</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func BlobFromImage(img *cv.Mat, scale float64, mean []float64, swapRB bool) *Tensor` | BlobFromImage converts a `cv.Mat` into a 4-D NCHW input blob suitable for a `Net`. |
| `func NewTensor(shape ...int) *Tensor` | NewTensor allocates a zero-filled Tensor with the given shape. |
| `func NewTensorFrom(shape []int, data []float64) *Tensor` | NewTensorFrom wraps data in a Tensor of the given shape. |
| `func (t *Tensor) At(index ...int) float64` | At returns the element at the given multi-index. |
| `func (t *Tensor) Clone() *Tensor` | Clone returns a deep copy with its own backing storage. |
| `func (t *Tensor) Dim(i int) int` | Dim returns the size of axis i. |
| `func (t *Tensor) Dims() int` | Dims returns the number of axes (the rank) of the tensor. |
| `func (t *Tensor) Equal(other *Tensor) bool` | Equal reports whether t and other have the same shape and identical data. |
| `func (t *Tensor) Len() int` | Len returns the total number of elements (the product of the shape). |
| `func (t *Tensor) Offset(index ...int) int` | Offset converts a multi-index into the flat Data offset. |
| `func (t *Tensor) Reshape(shape ...int) *Tensor` | Reshape returns a view of the tensor with a new shape that shares the Data slice. |
| `func (t *Tensor) Set(value float64, index ...int)` | Set stores value at the given multi-index. |
| `func (t *Tensor) String() string` | String returns a compact, human-readable description of the tensor's shape, useful in diagnostics. |

</details>

<details>
<summary><code>Transpose</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewTranspose(axis1, axis2 int) *Transpose` | NewTranspose builds a Transpose swapping the two given axes. |
| `func (t *Transpose) Forward(inputs []*Tensor) []*Tensor` | Forward swaps the two configured axes of the single input tensor. |

</details>

<details>
<summary><code>Upsample</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewUpsampleBilinear(scale int) *Upsample` | NewUpsampleBilinear builds a bilinear Upsample with a common factor. |
| `func NewUpsampleNearest(scale int) *Upsample` | NewUpsampleNearest builds a nearest-neighbour Upsample with a common factor. |
| `func (u *Upsample) Forward(inputs []*Tensor) []*Tensor` | Forward upsamples the single NCHW input tensor. |

</details>

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/dnn).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
