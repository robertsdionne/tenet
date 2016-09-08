package ten

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestSoftmaxGradient(t *testing.T) {
	x, dy := New(8), New(8)

	x.Data = []float64{
		0, 1, 2, 3, 4, 5, 6, 7,
	}

	dy.Data = []float64{
		0 - 0.00057661271,
		0 - 0.0015673959,
		1 - 0.0042606238,
		0 - 0.011581576,
		0 - 0.031481985,
		0 - 0.0855769,
		0 - 0.23262219,
		0 - 0.63233268,
	}

	y := Softmax(x)
	dx := SoftmaxGradient(dy, y)

	gradients := []float64{
		0.00026385227,
		0.00071567186,
		0.0061945468,
		0.00517216,
		0.013432883,
		0.031885084,
		0.052466642,
		-0.11013085,
	}

	assert.InDeltaSlice(t, gradients, dx.Data, 1e-7)
}
