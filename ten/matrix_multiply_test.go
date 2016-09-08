package ten

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestMatrixMultiplyGradient(t *testing.T) {
	w, x, dy := New(3, 5), New(5, 4), New(3, 4)

	w.Data = []float64{
		0, 1, 2, 3, 4,
		1, 2, 3, 4, 5,
		2, 3, 4, 5, 6,
	}

	x.Data = []float64{
		0, 1, 2, 3,
		1, 2, 3, 4,
		2, 3, 4, 5,
		3, 4, 5, 6,
		4, 5, 6, 7,
	}

	dy.Data = []float64{
		1, 1, 1, 1,
		1, 1, 1, 1,
		1, 1, 1, 1,
	}

	dw, dx := MatrixMultiplyGradient(dy, w, x)

	matrixMultiply := func(tensors ...Tensor) (y Tensor) {
		y = DualMatrixMultiply(tensors[0], tensors[1])
		return
	}

	gradients := TestGradients(matrixMultiply, dy, &w, &x)

	assert.Equal(t, gradients[&w], dw)
	assert.Equal(t, gradients[&x], dx)
}
