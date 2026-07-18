package stitch

// ChainHomographies composes a sequence of pairwise homographies into a single
// transform. hs[k] is expected to map image k+1 into the frame of image k; the
// returned homography maps the last image directly into the frame of the first.
// An empty slice yields the identity.
func ChainHomographies(hs []Homography) Homography {
	acc := IdentityHomography()
	for _, h := range hs {
		acc = acc.Mul(h)
	}
	return acc
}

// GlobalTransforms turns a chain of pairwise homographies into a
// globally-consistent set of transforms into a common reference frame, the
// bundle-free alignment used by incremental panorama stitchers. pairwise[k] maps
// image k+1 into the frame of image k, so pairwise has one fewer element than
// the number of images. The returned slice has one homography per image (length
// len(pairwise)+1); result[i] maps image i into the frame of the reference
// image, chosen as image reference. reference must be a valid image index.
//
// Errors accumulate along the chain — the further an image is from the reference,
// the more pairwise transforms compound — which is exactly the drift that bundle
// adjustment later removes.
func GlobalTransforms(pairwise []Homography, reference int) []Homography {
	n := len(pairwise) + 1
	out := make([]Homography, n)
	if reference < 0 {
		reference = 0
	}
	if reference >= n {
		reference = n - 1
	}
	out[reference] = IdentityHomography()
	// Walk forward from the reference: image i+1 -> reference frame is
	// T(i) · pairwise[i], since pairwise[i] maps i+1 into i's frame.
	for i := reference; i < n-1; i++ {
		out[i+1] = out[i].Mul(pairwise[i])
	}
	// Walk backward from the reference: image i-1 -> reference frame is
	// T(i) · pairwise[i-1]⁻¹, since pairwise[i-1] maps i into (i-1)'s frame.
	for i := reference; i > 0; i-- {
		inv, ok := pairwise[i-1].Inverse()
		if !ok {
			inv = IdentityHomography()
		}
		out[i-1] = out[i].Mul(inv)
	}
	return out
}

// IncrementalAligner builds a globally-consistent chain of transforms one image
// at a time, without a global bundle adjustment. The first image added defines
// the reference frame (identity); each subsequent image is registered by the
// homography that maps it into the frame of the previously-added image, and the
// aligner composes that with the running transform to place the new image in the
// reference frame.
type IncrementalAligner struct {
	globals []Homography
}

// NewIncrementalAligner returns an empty aligner. The first call to
// [IncrementalAligner.Add] (whose homography is ignored) fixes the reference
// frame.
func NewIncrementalAligner() *IncrementalAligner {
	return &IncrementalAligner{}
}

// Add registers the next image. relToPrev is the homography mapping the new
// image into the frame of the most recently added image; it is ignored for the
// very first image, which defines the reference frame. Add returns the global
// transform (into the reference frame) assigned to the newly added image.
func (a *IncrementalAligner) Add(relToPrev Homography) Homography {
	if len(a.globals) == 0 {
		g := IdentityHomography()
		a.globals = append(a.globals, g)
		return g
	}
	prev := a.globals[len(a.globals)-1]
	g := prev.Mul(relToPrev)
	a.globals = append(a.globals, g)
	return g
}

// Count returns the number of images registered so far.
func (a *IncrementalAligner) Count() int {
	return len(a.globals)
}

// Transform returns the global transform of image i (into the reference frame)
// and true, or the identity and false if i is out of range.
func (a *IncrementalAligner) Transform(i int) (Homography, bool) {
	if i < 0 || i >= len(a.globals) {
		return IdentityHomography(), false
	}
	return a.globals[i], true
}

// Transforms returns a copy of every global transform accumulated so far, indexed
// by the order images were added.
func (a *IncrementalAligner) Transforms() []Homography {
	out := make([]Homography, len(a.globals))
	copy(out, a.globals)
	return out
}
