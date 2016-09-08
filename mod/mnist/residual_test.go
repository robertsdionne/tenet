package mnist

import (
	"github.com/robertsdionne/tenet/ten"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestResidualGradient(t *testing.T) {
	W, b, x, dy := ten.New(3, 3), ten.New(3, 1), ten.New(3, 1), ten.New(3, 1)

	W.Data = []float64{
		0, 1, 2,
		1, 2, 3,
		2, 3, 4,
	}

	b.Data = []float64{
		0,
		1,
		2,
	}

	x.Data = []float64{
		0,
		1,
		2,
	}

	dy.Data = []float64{
		1,
		1,
		1,
	}

	_, h1, _ := residualForward(W, b, x)
	dw, db, dx, _, _ := residualGradient(W, b, x, h1, dy)

	forward := func(tensors ...ten.Tensor) (y ten.Tensor) {
		_, _, y = dualResidualForward(tensors[0], tensors[1], tensors[2])
		return
	}

	gradients := ten.TestGradients(forward, dy, &W, &b, &x)

	assert.Equal(t, gradients[&W], dw)
	assert.Equal(t, gradients[&b], db)
	assert.Equal(t, gradients[&x], dx)
}
