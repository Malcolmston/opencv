package dnn

import "fmt"

// Layer is a single stage of a neural network. It maps a slice of input
// tensors to a slice of output tensors. Most layers are single-input,
// single-output; multi-input layers such as [Concat] and [Add] read every
// element of inputs. Implementations must not retain or mutate their inputs —
// they return freshly allocated tensors — so a layer may be reused across
// calls and is safe for repeated inference.
type Layer interface {
	Forward(inputs []*Tensor) []*Tensor
}

// namedLayer pairs a layer with the name it was registered under inside a Net.
type namedLayer struct {
	name  string
	layer Layer
}

// Net is a feed-forward network: an ordered list of named [Layer]s evaluated
// in sequence. The single output of each layer becomes the single input of the
// next, so a Net models a linear (non-branching) pipeline such as
// Conv → ReLU → Pool → Flatten → Dense → Softmax. Layers may be looked up by
// name with [Net.LayerByName].
//
// Build a Net directly with [NewNet] and [Net.Add]/[Net.AddNamed], or fluently
// with a [Sequential] builder.
type Net struct {
	layers []namedLayer
}

// NewNet returns an empty network.
func NewNet() *Net { return &Net{} }

// Add appends layer with an automatically generated name ("layer0", "layer1",
// …) and returns the receiver so calls can be chained.
func (n *Net) Add(layer Layer) *Net {
	return n.AddNamed(fmt.Sprintf("layer%d", len(n.layers)), layer)
}

// AddNamed appends layer under the given name and returns the receiver. It
// panics if the name is empty or already used.
func (n *Net) AddNamed(name string, layer Layer) *Net {
	if name == "" {
		panic("dnn: layer name must not be empty")
	}
	if layer == nil {
		panic("dnn: cannot add a nil layer")
	}
	for _, l := range n.layers {
		if l.name == name {
			panic(fmt.Sprintf("dnn: duplicate layer name %q", name))
		}
	}
	n.layers = append(n.layers, namedLayer{name: name, layer: layer})
	return n
}

// Len returns the number of layers in the network.
func (n *Net) Len() int { return len(n.layers) }

// LayerByName returns the layer registered under name and whether it was found.
func (n *Net) LayerByName(name string) (Layer, bool) {
	for _, l := range n.layers {
		if l.name == name {
			return l.layer, true
		}
	}
	return nil, false
}

// LayerNames returns the layer names in evaluation order.
func (n *Net) LayerNames() []string {
	names := make([]string, len(n.layers))
	for i, l := range n.layers {
		names[i] = l.name
	}
	return names
}

// Forward runs input through every layer in order and returns the final
// tensor. Each layer is invoked with the single output of its predecessor. It
// panics if the network is empty or any layer does not yield exactly one
// tensor (use [Net.ForwardMulti] for multi-output layers).
func (n *Net) Forward(input *Tensor) *Tensor {
	if len(n.layers) == 0 {
		panic("dnn: Forward on an empty network")
	}
	cur := []*Tensor{input}
	for _, l := range n.layers {
		cur = l.layer.Forward(cur)
		if len(cur) != 1 {
			panic(fmt.Sprintf("dnn: layer %q produced %d outputs, sequential Forward requires 1", l.name, len(cur)))
		}
	}
	return cur[0]
}

// ForwardMulti runs inputs through the network, passing the full output slice
// of each layer as the input slice of the next. It returns the final layer's
// outputs. It panics on an empty network.
func (n *Net) ForwardMulti(inputs []*Tensor) []*Tensor {
	if len(n.layers) == 0 {
		panic("dnn: ForwardMulti on an empty network")
	}
	cur := inputs
	for _, l := range n.layers {
		cur = l.layer.Forward(cur)
	}
	return cur
}

// Sequential is a fluent builder for a linear [Net]. Each method appends a
// layer and returns the builder, so a network can be assembled in one
// expression; call [Sequential.Build] to obtain the finished Net.
type Sequential struct {
	net *Net
}

// NewSequential returns an empty Sequential builder.
func NewSequential() *Sequential { return &Sequential{net: NewNet()} }

// Add appends an arbitrary layer with an auto-generated name.
func (s *Sequential) Add(layer Layer) *Sequential {
	s.net.Add(layer)
	return s
}

// Named appends an arbitrary layer under an explicit name.
func (s *Sequential) Named(name string, layer Layer) *Sequential {
	s.net.AddNamed(name, layer)
	return s
}

// Conv2D appends a 2-D convolution layer.
func (s *Sequential) Conv2D(c *Conv2D) *Sequential { return s.Add(c) }

// MaxPool2D appends a max-pooling layer.
func (s *Sequential) MaxPool2D(p *MaxPool2D) *Sequential { return s.Add(p) }

// AvgPool2D appends an average-pooling layer.
func (s *Sequential) AvgPool2D(p *AvgPool2D) *Sequential { return s.Add(p) }

// ReLU appends a rectified-linear activation.
func (s *Sequential) ReLU() *Sequential { return s.Add(&ReLU{}) }

// LeakyReLU appends a leaky rectified-linear activation with the given slope
// for negative inputs.
func (s *Sequential) LeakyReLU(alpha float64) *Sequential {
	return s.Add(&LeakyReLU{Alpha: alpha})
}

// Sigmoid appends a logistic-sigmoid activation.
func (s *Sequential) Sigmoid() *Sequential { return s.Add(&Sigmoid{}) }

// Tanh appends a hyperbolic-tangent activation.
func (s *Sequential) Tanh() *Sequential { return s.Add(&Tanh{}) }

// Flatten appends a layer that flattens each batch element to a vector.
func (s *Sequential) Flatten() *Sequential { return s.Add(&Flatten{}) }

// Dense appends a fully-connected layer.
func (s *Sequential) Dense(d *FullyConnected) *Sequential { return s.Add(d) }

// BatchNorm appends a batch-normalization layer.
func (s *Sequential) BatchNorm(b *BatchNorm) *Sequential { return s.Add(b) }

// Softmax appends a softmax over the last axis.
func (s *Sequential) Softmax() *Sequential { return s.Add(NewSoftmax()) }

// Build finalizes and returns the assembled network.
func (s *Sequential) Build() *Net { return s.net }
