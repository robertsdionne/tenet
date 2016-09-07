package ten

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestSlice(t *testing.T) {
	tensor := New(4, 2, 2)
	tensor.Data = []float64{
		0, 1,
		1, 2,

		1, 2,
		2, 3,

		2, 3,
		3, 4,

		3, 4,
		4, 5,
	}

	assert.Equal(t, []float64{
		0, 1,
		1, 2,
	}, tensor.Slice(0).Data)

	assert.Equal(t, []float64{
		1, 2,
		2, 3,
	}, tensor.Slice(1).Data)

	assert.Equal(t, []float64{
		2, 3,
		3, 4,
	}, tensor.Slice(2).Data)

	assert.Equal(t, []float64{
		3, 4,
		4, 5,
	}, tensor.Slice(3).Data)

	replace := New(2, 2)
	replace.Data = []float64{
		3, 3,
		3, 3,
	}

	tensor.AssignSlice(0, replace)

	assert.Equal(t, []float64{
		3, 3,
		3, 3,
	}, tensor.Slice(0).Data)
}

func TestDualCopy(t *testing.T) {
	tensor := Uniform(-1, 1)(2, 2)
	dual := tensor.DualCopy()

	assert.Equal(t, tensor.Data, dual.Slice(0).Data)
}
