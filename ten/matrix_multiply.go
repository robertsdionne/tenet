package ten

// MatrixMultiply multiplies two matrices w and x.
func MatrixMultiply(w, x Tensor) (y Tensor) {
	m, r, n := int(w.Shape[0]), int(w.Shape[1]), int(x.Shape[1])

	y = New(int32(m), int32(n))

	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			for k := 0; k < r; k++ {
				*y.At(i, j) += *w.At(i, k) * *x.At(k, j)
			}
		}
	}

	return
}

func DualMatrixMultiply(w, x Tensor) (y Tensor) {
	m, n := int(w.Shape[1]), int(x.Shape[2])

	y = New(2, int32(m), int32(n))

	y.AssignSlice(0, MatrixMultiply(w.Slice(0), x.Slice(0)))
	y.AssignSlice(1, Add(MatrixMultiply(w.Slice(0), x.Slice(1)), MatrixMultiply(w.Slice(1), x.Slice(0))))

	return
}

// y = w * x
// y_ij = w_ik * x_kj

// dw_ik = dy_ij * x_kj
// dx_kj = dy_ij * w_ik

// y => dw
// i => i
// j => k
// w => dy
// i => i
// k => j
// x => x
// k => k
// j => j

// y => dx
// i => k
// j => j
// w => dy
// i => i
// k => j
// x => w
// k => i
// j => k

func MatrixMultiplyGradient(dy, w, x Tensor) (dw, dx Tensor) {
	dw, dx = NewLike(w), NewLike(x)

	m, r, n := int(w.Shape[0]), int(x.Shape[0]), int(x.Shape[1])

	for i := 0; i < m; i++ {
		for k := 0; k < r; k++ {
			for j := 0; j < n; j++ {
				*dw.At(i, k) += *dy.At(i, j) * *x.At(k, j)
			}
		}
	}

	for k := 0; k < r; k++ {
		for j := 0; j < n; j++ {
			for i := 0; i < m; i++ {
				*dx.At(k, j) += *dy.At(i, j) * *w.At(i, k)
			}
		}
	}

	return
}
