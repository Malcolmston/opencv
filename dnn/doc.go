// Package dnn is a small, standard-library-only feed-forward neural-network
// inference engine, a from-scratch port of a useful subset of OpenCV's dnn
// module. It runs pre-trained convolutional and fully-connected networks that
// you assemble and weight by hand in Go — there is no model-file parsing, no
// autodiff and no training.
//
// The package builds on the root cv package (github.com/malcolmston/opencv)
// only at its image boundary: [BlobFromImage] turns a [cv.Mat] into a network
// input and [BlobToImage] turns an output back into a Mat. Everything in
// between is plain Go and the standard library (math). It uses no cgo and no
// third-party code, and it does not import the other cv/* subpackages.
//
// # Tensors
//
// A [Tensor] is a row-major n-dimensional array of float64 samples with an
// integer shape. Image data uses the NCHW convention — axis 0 batch, axis 1
// channel, then height and width — which is what [BlobFromImage] produces and
// what [Conv2D] and the pooling layers expect. Feature matrices fed to a dense
// layer are rank 2, [batch, features].
//
// # Layers and networks
//
// A [Layer] maps a slice of input tensors to a slice of output tensors via its
// single method Forward(inputs []*Tensor) []*Tensor. The provided layers are:
//
//   - [Conv2D] — 2-D convolution with stride, zero padding and dilation.
//   - [ConvTranspose2D] — transposed convolution (deconvolution) for upsampling.
//   - [MaxPool2D], [AvgPool2D] — spatial downsampling.
//   - [GlobalAvgPool] — collapse each channel's spatial extent to a scalar.
//   - [ReLU], [LeakyReLU], [PReLU], [ELU], [Sigmoid], [Tanh], [Mish], [Swish]
//     (SiLU) — elementwise activations.
//   - [FullyConnected] (aliased [Dense]) — inner-product layer.
//   - [Softmax] — probability normalization along an axis.
//   - [BatchNorm] — inference-time per-channel normalization.
//   - [LRN] — local response normalization (across- or within-channel).
//   - [Dropout] — inference-time passthrough (training-only regularizer).
//   - [Flatten], [Reshape] — reinterpret the shape of a tensor.
//   - [Permute], [Transpose] — reorder axes, moving data.
//   - [Slice] — extract a strided sub-range along one axis.
//   - [Padding] — enlarge a tensor with a constant-valued border.
//   - [Upsample] — nearest or bilinear spatial magnification.
//   - [ArgMax] — reduce an axis to the index of its maximum.
//   - [Concat] — join tensors along an axis.
//   - [Add] — elementwise residual sum.
//   - [Eltwise] — elementwise sum/product/max reduction across inputs.
//
// Two free helpers support detection and classification post-processing:
// [NMSBoxes] applies greedy non-maximum suppression to scored [Box]es, and
// [ClassifyTopK] returns the highest-scoring classes of an output vector.
//
// A [Net] chains layers into a linear pipeline: [Net.Forward] feeds the output
// of each layer into the next and returns the final tensor. Assemble one
// directly with [NewNet] and [Net.Add], or fluently with a [Sequential]
// builder:
//
//	net := dnn.NewSequential().
//		Conv2D(dnn.NewConv2D(weights, bias, 1, 0, 1)).
//		ReLU().
//		MaxPool2D(dnn.NewMaxPool2D(2, 2)).
//		Flatten().
//		Dense(dnn.NewFullyConnected(fcW, fcB)).
//		Softmax().
//		Build()
//	probs := net.Forward(dnn.BlobFromImage(img, 1.0/255, nil, false))
//
// Weights are supplied as tensors of a documented shape: Conv2D expects
// [outC, inC, kH, kW], FullyConnected expects [outFeatures, inFeatures], and
// biases are rank-1 tensors sized to the output. Because there is no training,
// you set these values yourself, whether hand-computed, exported from another
// tool, or read from your own format.
//
// # Determinism
//
// All arithmetic is deterministic float64 with a fixed evaluation order, so a
// given network and input always produce identical output. There is no hidden
// global state and no randomness.
//
// # Errors and panics
//
// Following the root package's convention, functions panic on programmer
// errors — mismatched shapes, wrong ranks, non-positive dimensions — rather
// than returning errors, mirroring a Go slice index out of range. Panic
// messages are prefixed "dnn:".
//
// # Scope
//
// Deliberately out of scope: parsing ONNX, Caffe, TensorFlow or other model
// files; GPU or SIMD acceleration; recurrent, attention or transformer layers;
// float32 storage; and any form of automatic differentiation or training.
package dnn
