// Package core provides the small fixed-size value types from OpenCV's core
// module — the concrete point, vector, matrix, size, rectangle and helper
// types that cv2 exposes as templated typedefs (cv::Point_, cv::Point3_,
// cv::Size_, cv::Rect_, cv::Vec, cv::Matx, cv::Scalar, cv::Complex, cv::Range,
// cv::RotatedRect, cv::KeyPoint, cv::DMatch and cv::TermCriteria).
//
// The root cv package carries a handful of these (Point, Rect, Scalar) tuned
// for image drawing; this package supplies the full family of numeric variants
// with their arithmetic, so numerical and geometric code can port from OpenCV
// with the same type names and operations. Everything here is written against
// the Go standard library only — no cgo and no third-party dependencies — and
// is deterministic and allocation-light: the vector and matrix types are plain
// fixed-length arrays passed by value.
//
// # Naming
//
// Type suffixes follow OpenCV: b is uint8, s is int16, w is uint16, i is int,
// f is float32 and d is float64. So Vec3b is three bytes, Point2f is a
// float32 point and Matx33d is a 3×3 float64 matrix.
package core
