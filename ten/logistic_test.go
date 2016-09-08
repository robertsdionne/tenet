package ten

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestLogisticGradient(t *testing.T) {
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

	y := Logistic(x)
	dx := LogisticGradient(dy, y)

	logistic := func(tensors ...Tensor) (y Tensor) {
		y = DualLogistic(tensors[0])
		return
	}

	gradients := TestGradients(logistic, &x)

	assert.InDeltaSlice(t, gradients[&x].Data, dx.Data, 1e-10)
}
