package cudacore

// Stream is a CPU-backed stand-in for cv::cuda::Stream. Every operation in this
// package runs synchronously, so a Stream carries no state and enqueues no work.
// It exists solely so that code written against the CUDA API — which threads an
// optional stream through calls — ports without change.
type Stream struct{}

// NewStream returns a ready-to-use no-op Stream.
func NewStream() *Stream {
	return &Stream{}
}

// WaitForCompletion returns immediately, the analogue of
// cv::cuda::Stream::waitForCompletion. Because operations complete synchronously
// before they return, there is never any outstanding work to wait for.
func (s *Stream) WaitForCompletion() {}

// QueryIfComplete reports whether all work on the stream has finished, the
// analogue of cv::cuda::Stream::queryIfComplete. Work is always synchronous, so
// it is always true.
func (s *Stream) QueryIfComplete() bool {
	return true
}

// Event is a CPU-backed stand-in for cv::cuda::Event. With no asynchronous
// device work to mark, recording an Event and waiting on it are no-ops, and the
// elapsed time between two Events is always zero.
type Event struct{}

// NewEvent returns a ready-to-use no-op Event.
func NewEvent() *Event {
	return &Event{}
}

// Record marks the Event in the given stream, the analogue of
// cv::cuda::Event::record. There is no asynchronous work to timestamp, so it is
// a no-op; the stream argument is accepted for API compatibility.
func (e *Event) Record(_ *Stream) {}

// WaitForCompletion returns immediately, the analogue of
// cv::cuda::Event::waitForCompletion. A no-op Event is always already complete.
func (e *Event) WaitForCompletion() {}

// QueryIfComplete reports whether the Event has been reached, the analogue of
// cv::cuda::Event::queryIfComplete. It is always true.
func (e *Event) QueryIfComplete() bool {
	return true
}

// ElapsedTime returns the elapsed time in milliseconds between two recorded
// Events, the analogue of cv::cuda::Event::elapsedTime. Because nothing is timed
// here, it always returns 0.
func ElapsedTime(_, _ *Event) float64 {
	return 0
}

// StreamWaitEvent makes stream wait for event before proceeding, the analogue of
// cv::cuda::Stream::waitEvent. Both are no-ops here, so it does nothing; the
// arguments are accepted for API compatibility.
func StreamWaitEvent(_ *Stream, _ *Event) {}
