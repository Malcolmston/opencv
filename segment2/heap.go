package segment2

// segment2cand is a candidate pixel for seeded region growing, ordered by its
// colour difference from a region mean with a FIFO tie-break for determinism.
type segment2cand struct {
	index int
	label int
	diff  float64
	order int
}

// segment2candHeap is a deterministic min-heap of region-growing candidates,
// keyed on diff then insertion order.
type segment2candHeap struct {
	items []segment2cand
	order int
}

func (h *segment2candHeap) less(i, j int) bool {
	if h.items[i].diff != h.items[j].diff {
		return h.items[i].diff < h.items[j].diff
	}
	return h.items[i].order < h.items[j].order
}

// Len reports the number of queued candidates.
func (h *segment2candHeap) Len() int { return len(h.items) }

func (h *segment2candHeap) push(c segment2cand) {
	c.order = h.order
	h.order++
	h.items = append(h.items, c)
	i := len(h.items) - 1
	for i > 0 {
		parent := (i - 1) / 2
		if !h.less(i, parent) {
			break
		}
		h.items[i], h.items[parent] = h.items[parent], h.items[i]
		i = parent
	}
}

func (h *segment2candHeap) pop() segment2cand {
	n := len(h.items)
	top := h.items[0]
	h.items[0] = h.items[n-1]
	h.items = h.items[:n-1]
	n--
	i := 0
	for {
		l := 2*i + 1
		r := 2*i + 2
		smallest := i
		if l < n && h.less(l, smallest) {
			smallest = l
		}
		if r < n && h.less(r, smallest) {
			smallest = r
		}
		if smallest == i {
			break
		}
		h.items[i], h.items[smallest] = h.items[smallest], h.items[i]
		i = smallest
	}
	return top
}
