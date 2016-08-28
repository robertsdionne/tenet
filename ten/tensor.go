package ten

import (
	"github.com/robertsdionne/tenet/prot"
	"github.com/robertsdionne/tenet/ten/size"
	"github.com/robertsdionne/tenet/ten/stride"
)

// Shape is the shape of a Tensor.
type Shape []int32

// Tensor is a multidimensional array of floating point values.
type Tensor prot.Tensor

// ShapeMap describes the shapes of a set of tensors.
type ShapeMap map[string]Shape

// TensorMap describes the values of a set of tensors.
type TensorMap map[string]Tensor

// New constructs a new Tensor.
func New(shape ...int32) (tensor Tensor) {
	return Tensor{
		Shape:  shape,
		Stride: stride.C(shape...),
		Data:   make([]float64, size.Of(shape...)),
	}
}

func NewLike(t0 Tensor) (t1 Tensor) {
	t1 = New(t0.Shape...)
	return
}

// At retrieves the component at the index.
func (tensor *Tensor) At(index ...int) (at *float64) {
	offset := 0
	for i, stride := range tensor.Stride {
		offset += index[i] * int(stride)
	}
	at = &tensor.Data[offset]
	return
}
