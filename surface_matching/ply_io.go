package surface_matching

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"strconv"
	"strings"
)

// WritePLY writes the cloud to path as an ASCII PLY file with x, y, z vertex
// coordinates and, when the cloud carries normals, nx, ny, nz. Coordinates are
// written as double-precision so a subsequent [ReadPLY] round-trips them
// exactly. It complements the read-only [LoadPLY] with the export side OpenCV's
// writePLY provides.
func WritePLY(path string, pc *PointCloud) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	if err := writePLYASCII(f, pc); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

// WritePLYBinary writes the cloud to path as a little-endian binary PLY file
// (format binary_little_endian 1.0) with double-precision x, y, z and, when
// present, nx, ny, nz. Binary is far more compact and faster to parse than ASCII
// for large clouds; [ReadPLY] auto-detects and reads it back.
func WritePLYBinary(path string, pc *PointCloud) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	if err := writePLYBinary(f, pc); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

// plyHeader emits the shared PLY header for the given format token, declaring
// the vertex element and its double-precision properties.
func plyHeader(w io.Writer, format string, n int, hasNormals bool) error {
	var b strings.Builder
	b.WriteString("ply\n")
	fmt.Fprintf(&b, "format %s 1.0\n", format)
	b.WriteString("comment written by github.com/malcolmston/opencv surface_matching\n")
	fmt.Fprintf(&b, "element vertex %d\n", n)
	b.WriteString("property double x\nproperty double y\nproperty double z\n")
	if hasNormals {
		b.WriteString("property double nx\nproperty double ny\nproperty double nz\n")
	}
	b.WriteString("end_header\n")
	_, err := io.WriteString(w, b.String())
	return err
}

// writePLYASCII renders the cloud as ASCII PLY to w.
func writePLYASCII(w io.Writer, pc *PointCloud) error {
	hasN := len(pc.Normals) == len(pc.Points)
	bw := bufio.NewWriter(w)
	if err := plyHeader(bw, "ascii", len(pc.Points), hasN); err != nil {
		return err
	}
	for i, p := range pc.Points {
		if hasN {
			n := pc.Normals[i]
			if _, err := fmt.Fprintf(bw, "%g %g %g %g %g %g\n", p[0], p[1], p[2], n[0], n[1], n[2]); err != nil {
				return err
			}
		} else {
			if _, err := fmt.Fprintf(bw, "%g %g %g\n", p[0], p[1], p[2]); err != nil {
				return err
			}
		}
	}
	return bw.Flush()
}

// writePLYBinary renders the cloud as little-endian binary PLY to w.
func writePLYBinary(w io.Writer, pc *PointCloud) error {
	hasN := len(pc.Normals) == len(pc.Points)
	bw := bufio.NewWriter(w)
	if err := plyHeader(bw, "binary_little_endian", len(pc.Points), hasN); err != nil {
		return err
	}
	var buf [8]byte
	put := func(v float64) error {
		binary.LittleEndian.PutUint64(buf[:], math.Float64bits(v))
		_, err := bw.Write(buf[:])
		return err
	}
	for i, p := range pc.Points {
		for k := 0; k < 3; k++ {
			if err := put(p[k]); err != nil {
				return err
			}
		}
		if hasN {
			n := pc.Normals[i]
			for k := 0; k < 3; k++ {
				if err := put(n[k]); err != nil {
					return err
				}
			}
		}
	}
	return bw.Flush()
}

// ReadPLY reads a PLY file in either ASCII or binary (little- or big-endian)
// format, auto-detecting from the header. It parses the vertex element's x, y, z
// coordinates and, when present, nx, ny, nz normals, decoding every standard PLY
// scalar property type (char/uchar/short/ushort/int/uint/float/double and their
// intN aliases) and skipping unrecognised or non-coordinate properties. It
// assumes the vertex element appears first, as writers in this package and
// OpenCV emit.
//
// Unlike [LoadPLY], which is ASCII-only, ReadPLY handles the binary files
// written by [WritePLYBinary]. If the file carries no normals the returned
// cloud's Normals slice is empty.
func ReadPLY(path string) (*PointCloud, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return readPLY(f)
}

// plyProperty describes one scalar vertex property parsed from the header.
type plyProperty struct {
	name  string
	size  int  // encoded byte width
	float bool // true for float/double, false for integer types
}

// plyTypeInfo maps a PLY scalar type name to its byte width and float-ness.
func plyTypeInfo(t string) (size int, isFloat bool, ok bool) {
	switch t {
	case "char", "int8", "uchar", "uint8":
		return 1, false, true
	case "short", "int16", "ushort", "uint16":
		return 2, false, true
	case "int", "int32", "uint", "uint32":
		return 4, false, true
	case "float", "float32":
		return 4, true, true
	case "double", "float64":
		return 8, true, true
	}
	return 0, false, false
}

// readPLY is the format-detecting reader behind [ReadPLY], split out so it can
// be exercised on an in-memory stream.
func readPLY(r io.Reader) (*PointCloud, error) {
	br := bufio.NewReader(r)
	line, err := readHeaderLine(br)
	if err != nil || line != "ply" {
		return nil, fmt.Errorf("surface_matching: not a PLY file")
	}
	var (
		format      string
		vertexCount int
		inVertex    bool
		props       []plyProperty
	)
	for {
		line, err = readHeaderLine(br)
		if err != nil {
			return nil, err
		}
		fields := strings.Fields(line)
		if len(fields) == 0 {
			continue
		}
		switch fields[0] {
		case "format":
			if len(fields) >= 2 {
				format = fields[1]
			}
		case "comment":
		case "element":
			if len(fields) >= 3 && fields[1] == "vertex" {
				inVertex = true
				vertexCount, _ = strconv.Atoi(fields[2])
			} else {
				inVertex = false
			}
		case "property":
			if inVertex && len(fields) >= 3 && fields[1] != "list" {
				size, isFloat, ok := plyTypeInfo(fields[1])
				if !ok {
					return nil, fmt.Errorf("surface_matching: unsupported PLY property type %q", fields[1])
				}
				props = append(props, plyProperty{name: fields[len(fields)-1], size: size, float: isFloat})
			}
		case "end_header":
			goto body
		}
	}
body:
	col := map[string]int{}
	for i, p := range props {
		col[p.name] = i
	}
	need := func(name string) (int, bool) { i, ok := col[name]; return i, ok }
	xi, xok := need("x")
	yi, yok := need("y")
	zi, zok := need("z")
	if !xok || !yok || !zok {
		return nil, fmt.Errorf("surface_matching: PLY vertex is missing x/y/z properties")
	}
	nxi, nxok := need("nx")
	nyi, nyok := need("ny")
	nzi, nzok := need("nz")
	hasNormals := nxok && nyok && nzok

	switch format {
	case "", "ascii":
		return readPLYASCII(br, props, vertexCount, xi, yi, zi, nxi, nyi, nzi, hasNormals)
	case "binary_little_endian":
		return readPLYBinary(br, binary.LittleEndian, props, vertexCount, xi, yi, zi, nxi, nyi, nzi, hasNormals)
	case "binary_big_endian":
		return readPLYBinary(br, binary.BigEndian, props, vertexCount, xi, yi, zi, nxi, nyi, nzi, hasNormals)
	default:
		return nil, fmt.Errorf("surface_matching: unsupported PLY format %q", format)
	}
}

// readHeaderLine reads one CR/LF-terminated header line from br.
func readHeaderLine(br *bufio.Reader) (string, error) {
	s, err := br.ReadString('\n')
	if err != nil && s == "" {
		return "", err
	}
	return strings.TrimRight(s, "\r\n"), nil
}

// readPLYASCII parses vertexCount whitespace-delimited vertex rows from br.
func readPLYASCII(br *bufio.Reader, props []plyProperty, vertexCount, xi, yi, zi, nxi, nyi, nzi int, hasNormals bool) (*PointCloud, error) {
	sc := bufio.NewScanner(br)
	sc.Buffer(make([]byte, 0, 1<<16), 1<<26)
	pc := &PointCloud{}
	for i := 0; i < vertexCount; i++ {
		if !sc.Scan() {
			return nil, fmt.Errorf("surface_matching: PLY ended before %d vertices", vertexCount)
		}
		f := strings.Fields(sc.Text())
		if len(f) < len(props) {
			return nil, fmt.Errorf("surface_matching: PLY vertex line %d has too few fields", i)
		}
		p, err := parseVec(f, xi, yi, zi)
		if err != nil {
			return nil, err
		}
		pc.Points = append(pc.Points, p)
		if hasNormals {
			n, err := parseVec(f, nxi, nyi, nzi)
			if err != nil {
				return nil, err
			}
			pc.Normals = append(pc.Normals, normalize3(n))
		}
	}
	return pc, sc.Err()
}

// readPLYBinary parses vertexCount fixed-width binary vertex records from br
// using the given byte order.
func readPLYBinary(br *bufio.Reader, order binary.ByteOrder, props []plyProperty, vertexCount, xi, yi, zi, nxi, nyi, nzi int, hasNormals bool) (*PointCloud, error) {
	rowSize := 0
	for _, p := range props {
		rowSize += p.size
	}
	row := make([]byte, rowSize)
	pc := &PointCloud{}
	for i := 0; i < vertexCount; i++ {
		if _, err := io.ReadFull(br, row); err != nil {
			return nil, fmt.Errorf("surface_matching: PLY binary vertex %d: %w", i, err)
		}
		vals := make([]float64, len(props))
		off := 0
		for j, p := range props {
			vals[j] = decodePLYScalar(row[off:off+p.size], p, order)
			off += p.size
		}
		pc.Points = append(pc.Points, Vec3{vals[xi], vals[yi], vals[zi]})
		if hasNormals {
			pc.Normals = append(pc.Normals, normalize3(Vec3{vals[nxi], vals[nyi], vals[nzi]}))
		}
	}
	return pc, nil
}

// decodePLYScalar decodes one binary property value to float64 according to its
// declared type and the file's byte order.
func decodePLYScalar(b []byte, p plyProperty, order binary.ByteOrder) float64 {
	switch {
	case p.float && p.size == 8:
		return math.Float64frombits(order.Uint64(b))
	case p.float && p.size == 4:
		return float64(math.Float32frombits(order.Uint32(b)))
	case p.size == 1:
		return float64(int8(b[0]))
	case p.size == 2:
		return float64(int16(order.Uint16(b)))
	case p.size == 4:
		return float64(int32(order.Uint32(b)))
	case p.size == 8:
		return float64(int64(order.Uint64(b)))
	}
	return 0
}
