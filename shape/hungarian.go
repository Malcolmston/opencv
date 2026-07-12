package shape

import "math"

// SolveAssignment solves the linear assignment problem: given a cost matrix in
// which cost[i][j] is the price of assigning row i to column j, it returns an
// assignment that minimises the total cost. The result is a slice with one entry
// per row; assignment[i] is the column matched to row i, or -1 when row i is left
// unmatched (which only happens when there are more rows than columns). total is
// the sum of the matched entries.
//
// The matrix need not be square. It is internally padded to a square matrix with
// a large sentinel cost so that, whenever it is possible, every real row is
// matched to a real column before any padding cell is used; padded matches are
// reported as -1 and excluded from total. The exact O(n³) Hungarian
// (Kuhn–Munkres) algorithm is used via the Jonker–Volgenant shortest-augmenting-
// path formulation, so the optimum is found, not approximated.
//
// It panics if cost is empty or ragged (rows of differing length).
func SolveAssignment(cost [][]float64) (assignment []int, total float64) {
	rows := len(cost)
	if rows == 0 {
		panic("shape: SolveAssignment on empty matrix")
	}
	cols := len(cost[0])
	for _, r := range cost {
		if len(r) != cols {
			panic("shape: SolveAssignment ragged cost matrix")
		}
	}
	if cols == 0 {
		panic("shape: SolveAssignment on empty matrix")
	}

	// A sentinel strictly larger than any real cost, so padding cells are only
	// chosen when a real match is impossible.
	big := 0.0
	for _, r := range cost {
		for _, v := range r {
			if a := math.Abs(v); a > big {
				big = a
			}
		}
	}
	big = (big + 1) * float64(rows+cols)

	n := rows
	if cols > n {
		n = cols
	}
	square := make([][]float64, n)
	for i := 0; i < n; i++ {
		square[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			if i < rows && j < cols {
				square[i][j] = cost[i][j]
			} else {
				square[i][j] = big
			}
		}
	}

	rowToCol := hungarianSquare(square)
	assignment = make([]int, rows)
	for i := 0; i < rows; i++ {
		j := rowToCol[i]
		if j >= 0 && j < cols {
			assignment[i] = j
			total += cost[i][j]
		} else {
			assignment[i] = -1
		}
	}
	return assignment, total
}

// hungarianSquare solves a square assignment problem and returns, for each row,
// the column it is matched to. It uses the potentials/shortest-path form of the
// Hungarian algorithm (Jonker–Volgenant), which runs in O(n³).
func hungarianSquare(cost [][]float64) []int {
	n := len(cost)
	const inf = math.MaxFloat64
	u := make([]float64, n+1)
	v := make([]float64, n+1)
	p := make([]int, n+1) // p[j] = row assigned to column j (1-based; 0 = none)
	way := make([]int, n+1)
	for i := 1; i <= n; i++ {
		p[0] = i
		j0 := 0
		minv := make([]float64, n+1)
		used := make([]bool, n+1)
		for j := 0; j <= n; j++ {
			minv[j] = inf
		}
		for {
			used[j0] = true
			i0 := p[j0]
			delta := inf
			j1 := -1
			for j := 1; j <= n; j++ {
				if used[j] {
					continue
				}
				cur := cost[i0-1][j-1] - u[i0] - v[j]
				if cur < minv[j] {
					minv[j] = cur
					way[j] = j0
				}
				if minv[j] < delta {
					delta = minv[j]
					j1 = j
				}
			}
			for j := 0; j <= n; j++ {
				if used[j] {
					u[p[j]] += delta
					v[j] -= delta
				} else {
					minv[j] -= delta
				}
			}
			j0 = j1
			if p[j0] == 0 {
				break
			}
		}
		for {
			j1 := way[j0]
			p[j0] = p[j1]
			j0 = j1
			if j0 == 0 {
				break
			}
		}
	}
	result := make([]int, n)
	for i := range result {
		result[i] = -1
	}
	for j := 1; j <= n; j++ {
		if p[j] >= 1 {
			result[p[j]-1] = j - 1
		}
	}
	return result
}
