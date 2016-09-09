package mnist

import (
	"github.com/robertsdionne/tenet/ten"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestSoftmaxGradient(t *testing.T) {
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
		0,
		0,
	}

	_, _, y := softmaxForward(W, b, x)
	dw, db, dx, _, _ := softmaxGradient(W, b, x, y, dy)

	forward := func(tensors ...ten.Tensor) (y ten.Tensor) {
		_, _, y = dualSoftmaxForward(tensors[0], tensors[1], tensors[2])
		return
	}

	gradients := ten.TestGradients(forward, dy, &W, &b, &x)

	assert.InDeltaSlice(t, gradients[&W].Data, dw.Data, 1e-10)
	assert.InDeltaSlice(t, gradients[&b].Data, db.Data, 1e-10)
	assert.InDeltaSlice(t, gradients[&x].Data, dx.Data, 1e-10)
}
