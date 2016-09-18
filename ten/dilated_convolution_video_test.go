package ten

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestDilatedConvolutionVideoShape(t *testing.T) {
	x := NewTest(4, 3, 3, 2)
	w := NewTest(2, 3, 3, 2, 2)

	y := DilatedConvolutionVideo(w, x, 1)

	assert.Equal(t, []int32{4, 3, 3, 2}, y.Shape)
	assert.Equal(t, []float64{
		144, 176, 244, 304, 160, 208,
		244, 304, 390, 498, 244, 328,
		160, 208, 244, 328, 144, 208,

		208, 256, 328, 412, 208, 272,
		328, 412, 498, 642, 304, 412,
		208, 272, 304, 412, 176, 256,

		272, 336, 412, 520, 256, 336,
		412, 520, 606, 786, 364, 496,
		256, 336, 364, 496, 208, 304,

		132, 168, 194, 254, 116, 160,
		194, 254, 276, 375, 158, 230,
		116, 160, 158, 230, 84, 136,
	}, y.Data)
}

func TestDilatedConvolutionVideoGradient(t *testing.T) {
	x := NewTest(4, 3, 3, 2)
	w := NewTest(2, 3, 3, 2, 2)
	dy := Constant(1)(4, 3, 3, 2)

	dw, dx := DilatedConvolutionVideoGradient(w, x, dy, 2)

	dilatedConvolutionVideo := func(tensors ...Tensor) (y Tensor) {
		y = DualDilatedConvolutionVideo(tensors[0], tensors[1], 2)
		return
	}

	gradients := TestGradients(dilatedConvolutionVideo, dy, &w, &x)

	assert.Equal(t, gradients[&w], dw)
	assert.Equal(t, gradients[&x], dx)
}
