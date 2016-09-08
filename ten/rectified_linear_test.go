package ten

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestRectifiedLinearGradient(t *testing.T) {
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

	dx := RectifiedLinearGradient(dy, x)

	rectifiedLinear := func(tensors ...Tensor) (y Tensor) {
		y = DualRectifiedLinear(tensors[0])
		return
	}

	gradients := TestGradients(rectifiedLinear, &x)

	assert.InDeltaSlice(t, gradients[&x].Data, dx.Data, 1e-10)
}
