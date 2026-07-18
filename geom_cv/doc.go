// Package geom_cv provides computational-geometry primitives that are useful
// for computer vision, built entirely on the Go standard library and on the
// core types of the parent github.com/malcolmston/opencv package.
//
// The package deliberately reuses the parent library's geometry and image
// types rather than inventing incompatible ones: planar points are
// [github.com/malcolmston/opencv.Point2f] (floating point) and
// [github.com/malcolmston/opencv.Point] (integer pixel), rectangles are
// [github.com/malcolmston/opencv.Rect] and
// [github.com/malcolmston/opencv.RotatedRect], and raster output is written
// into [github.com/malcolmston/opencv.Mat]. Functions here therefore compose
// directly with the rest of the library.
//
// # Coordinate conventions
//
// Points live in image coordinates, where X is the column and Y is the row and
// Y increases downward. Signed areas and orientations follow the standard
// mathematical cross-product convention: [SignedArea] and [Orientation] are
// positive for a counter-clockwise winding in a right-handed frame. Because the
// image Y axis points down, a counter-clockwise winding in the mathematical
// sense appears clockwise on screen; the doc comment of each function states
// the exact rule it uses so callers are never surprised.
//
// # Contents
//
// The package covers the classic toolbox of planar computational geometry as it
// is used in vision pipelines:
//
//   - vector primitives: [Add], [Sub], [Scale], [Dot], [Cross], [Norm],
//     [Distance], [Orientation], [Lerp] and friends;
//   - small value types: [Segment], [Line], [Circle], [Triangle] and
//     [BoundingBox], each with convenience methods;
//   - polygons: [PolygonArea], [PolygonCentroid], [PointInPolygon],
//     [WindingNumber], [IsConvex] and related predicates;
//   - convex hulls ([ConvexHull]) and rotating calipers ([ConvexDiameter],
//     [ConvexWidth], [AntipodalPairs], [MinAreaRect]);
//   - minimum enclosing shapes: [MinEnclosingCircle], [MinEnclosingBox];
//   - line and segment intersection and distance queries;
//   - Delaunay triangulation ([DelaunayTriangulation]) and its dual Voronoi
//     diagram ([VoronoiEdges], [VoronoiCells], [NearestSite]);
//   - polygon clipping ([ClipPolygon], [ClipSegmentToBox],
//     [PolygonIntersectionArea]);
//   - alpha shapes ([AlphaShapeEdges], [AlphaComplexTriangles]);
//   - rasterization into the library's image type ([PolygonMask],
//     [FillPolygon], [DrawPolygonOutline]).
//
// Everything is pure Go, CPU-only and deterministic: given the same input a
// function always returns the same result, which makes the package safe to use
// inside reproducible pipelines and easy to test against known answers.
package geom_cv
