package ten

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestNewTest(t *testing.T) {
	tensor := NewTest(3, 3, 3)

	assert.Equal(t, []float64{
		0, 1, 2,
		1, 2, 3,
		2, 3, 4,

		1, 2, 3,
		2, 3, 4,
		3, 4, 5,

		2, 3, 4,
		3, 4, 5,
		4, 5, 6,
	}, tensor.Data)
}
