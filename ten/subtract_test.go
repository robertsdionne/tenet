package ten

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestSubtractGradient(t *testing.T) {
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

	da, db := SubtractGradient(dc)

	subtract := func(tensors ...Tensor) (c Tensor) {
		c = DualSubtract(tensors[0], tensors[1])
		return
	}

	gradients := TestGradients(subtract, dc, &a, &b)

	assert.Equal(t, gradients[&a], da)
	assert.Equal(t, gradients[&b], db)
}
