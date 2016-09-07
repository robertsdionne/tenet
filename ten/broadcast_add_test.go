package ten

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestBroadcastAddGradient(t *testing.T) {
	x, b, dy := New(3, 3), New(3, 1), New(3, 3)

	x.Data = []float64{
		0, 1, 2,
		1, 2, 3,
		2, 3, 4,
	}

	b.Data = []float64{
		0,
		1,
		2,
	}

	dy.Data = []float64{
		1, 1, 1,
		1, 1, 1,
		1, 1, 1,
	}

	dx, db := BroadcastAddGradient(dy, b)

	broadcastAdd := func(tensors ...Tensor) (y Tensor) {
		y = DualBroadcastAdd(tensors[0], tensors[1])
		return
	}

	gradients := TestGradients(broadcastAdd, &x, &b)

	assert.Equal(t, gradients[&x], dx)
	assert.Equal(t, gradients[&b], db)
}
