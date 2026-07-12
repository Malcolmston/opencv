package cudabgsegm

// Stream is a CPU-backed no-op stand-in for OpenCV's cv::cuda::Stream. In real
// OpenCV a Stream queues asynchronous GPU work; here there is no device and no
// asynchrony, so a Stream carries no state and every operation on it returns
// immediately. It exists purely so that the CUDA API's habit of threading a
// Stream through each call compiles and runs unchanged.
//
// A nil *Stream is valid everywhere a Stream is accepted and is treated as the
// default (null) stream, mirroring cv::cuda::Stream::Null().
type Stream struct{}

// NewStream returns a new no-op Stream, mirroring the cv::cuda::Stream
// constructor.
func NewStream() *Stream {
	return &Stream{}
}

// WaitForCompletion returns immediately: there is no asynchronous work to wait
// for. It mirrors cv::cuda::Stream::waitForCompletion.
func (s *Stream) WaitForCompletion() {}

// QueryIfComplete always reports true, since work on this backend is synchronous
// and already finished by the time any call returns. It mirrors
// cv::cuda::Stream::queryIfComplete.
func (s *Stream) QueryIfComplete() bool { return true }
