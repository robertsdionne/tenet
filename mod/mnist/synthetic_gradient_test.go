package mnist

import (
	"github.com/robertsdionne/tenet/ten"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestSyntheticGradientGradient(t *testing.T) {
	W0, W1, b, y, label, loss := ten.New(3, 3), ten.New(3, 3), ten.New(3, 1), ten.New(3, 1), ten.New(3, 1), ten.New(3, 1)

	W0.Data = []float64{
		0, 1, 2,
		1, 2, 3,
		2, 3, 4,
	}

	W1.Data = []float64{
		0, 1, 2,
		1, 2, 3,
		2, 3, 4,
	}

	b.Data = []float64{
		0,
		1,
		2,
	}

	y.Data = []float64{
		0.5,
		0.25,
		0.25,
	}

	label.Data = []float64{
		0,
		1,
		0,
	}

	loss.Data = []float64{
		1,
		1,
		1,
	}

	_ = syntheticGradientForward(W0, W1, b, y, label)
	dw0, dw1, db := syntheticGradientGradient(W0, W1, b, y, label, loss)

	forward := func(tensors ...ten.Tensor) (dy ten.Tensor) {
		dy = dualSyntheticGradientForward(tensors[0], tensors[1], tensors[2], tensors[3], tensors[4])
		return
	}

	gradients := ten.TestGradients(forward, loss, &W0, &W1, &b, &y, &label)

	assert.Equal(t, gradients[&W0], dw0)
	assert.Equal(t, gradients[&W1], dw1)
	assert.Equal(t, gradients[&b], db)
}
