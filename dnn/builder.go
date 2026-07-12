package dnn

// This file extends the [Sequential] fluent builder with the single-input,
// single-output layers added alongside the original set. Multi-input layers
// (for example [Eltwise]) are intentionally omitted because a Sequential
// pipeline threads exactly one tensor between stages; add those with the
// generic [Sequential.Add].

// ConvTranspose2D appends a transposed-convolution (deconvolution) layer.
func (s *Sequential) ConvTranspose2D(c *ConvTranspose2D) *Sequential { return s.Add(c) }

// GlobalAvgPool appends a global average-pooling layer.
func (s *Sequential) GlobalAvgPool() *Sequential { return s.Add(&GlobalAvgPool{}) }

// LRN appends a local-response-normalization layer.
func (s *Sequential) LRN(l *LRN) *Sequential { return s.Add(l) }

// PReLU appends a parametric-ReLU activation with the given slope tensor.
func (s *Sequential) PReLU(slope *Tensor) *Sequential { return s.Add(NewPReLU(slope)) }

// ELU appends an exponential-linear-unit activation.
func (s *Sequential) ELU(alpha float64) *Sequential { return s.Add(&ELU{Alpha: alpha}) }

// Mish appends a Mish activation.
func (s *Sequential) Mish() *Sequential { return s.Add(&Mish{}) }

// Swish appends a Swish activation with the given beta.
func (s *Sequential) Swish(beta float64) *Sequential { return s.Add(&Swish{Beta: beta}) }

// SiLU appends a SiLU (Swish with beta 1) activation.
func (s *Sequential) SiLU() *Sequential { return s.Add(NewSiLU()) }

// Dropout appends an inference-time no-op dropout recording the given rate.
func (s *Sequential) Dropout(rate float64) *Sequential { return s.Add(&Dropout{Rate: rate}) }

// Reshape appends a layer that reshapes the input to the given shape.
func (s *Sequential) Reshape(shape ...int) *Sequential { return s.Add(NewReshape(shape...)) }

// Permute appends an axis-permutation layer.
func (s *Sequential) Permute(order ...int) *Sequential { return s.Add(NewPermute(order...)) }

// Transpose appends a two-axis transpose layer.
func (s *Sequential) Transpose(axis1, axis2 int) *Sequential {
	return s.Add(NewTranspose(axis1, axis2))
}

// Slice appends a single-axis slice layer.
func (s *Sequential) Slice(sl *Slice) *Sequential { return s.Add(sl) }

// Padding appends a constant-padding layer.
func (s *Sequential) Padding(p *Padding) *Sequential { return s.Add(p) }

// Upsample appends a spatial upsampling layer.
func (s *Sequential) Upsample(u *Upsample) *Sequential { return s.Add(u) }

// ArgMax appends an argmax-reduction layer over the given axis (keeping dims).
func (s *Sequential) ArgMax(axis int) *Sequential { return s.Add(NewArgMax(axis)) }
