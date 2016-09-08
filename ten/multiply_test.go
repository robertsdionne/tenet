package ten

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestMultiplyGradient(t *testing.T) {
	a, b, dc := New(3, 3), New(3, 3), New(3, 3)

	a.Data = []float64{
		0, 1, 2,
		1, 2, 3,
		2, 3, 4,
	}

	b.Data = []float64{
		0, 1, 2,
		1, 2, 3,
		2, 3, 4,
	}

	dc.Data = []float64{
		1, 1, 1,
		1, 1, 1,
		1, 1, 1,
	}

	da, db := MultiplyGradient(dc, a, b)

	multiply := func(tensors ...Tensor) (c Tensor) {
		c = DualMultiply(tensors[0], tensors[1])
		return
	}

	gradients := TestGradients(multiply, &a, &b)

	assert.InDeltaSlice(t, gradients[&a].Data, da.Data, 1e-10)
	assert.InDeltaSlice(t, gradients[&b].Data, db.Data, 1e-10)
}
