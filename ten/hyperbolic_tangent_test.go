package ten

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestHyperbolicTangentGradient(t *testing.T) {
	x, dy := New(3, 3), New(3, 3)

	x.Data = []float64{
		0, 1, 2,
		1, 2, 3,
		2, 3, 4,
	}

	dy.Data = []float64{
		1, 1, 1,
		1, 1, 1,
		1, 1, 1,
	}

	y := HyperbolicTangent(x)
	dx := HyperbolicTangentGradient(dy, y)

	hyperbolicTangent := func(tensors ...Tensor) (y Tensor) {
		y = DualHyperbolicTangent(tensors[0])
		return
	}

	gradients := TestGradients(hyperbolicTangent, dy, &x)

	assert.InDeltaSlice(t, gradients[&x].Data, dx.Data, 1e-10)
}
