package ten

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestDilatedConvolutionVideoShape(t *testing.T) {
	x := NewTest(4, 3, 3, 2)
	w := NewTest(2, 3, 3, 2, 2)

	y := DilatedConvolutionVideo(w, x, 1)

	assert.Equal(t, []int32{2, 3, 3, 2}, y.Shape)
	assert.Equal(t, []float64{
		144, 176, 244, 304, 160, 208,
		244, 304, 390, 498, 244, 328,
		160, 208, 244, 328, 144, 208,

		272, 336, 412, 520, 256, 336,
		412, 520, 606, 786, 364, 496,
		256, 336, 364, 496, 208, 304,
	}, y.Data)
}
