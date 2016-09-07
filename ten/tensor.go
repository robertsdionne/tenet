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

func (tensor *Tensor) Copy() (copy Tensor) {
	copy = NewLike(*tensor)

	for i, value := range tensor.Data {
		copy.Data[i] = value
	}

	return
}

func (tensor *Tensor) Slice(index int) (slice Tensor) {
	shape := tensor.Shape[1:]
	firstStride := int(tensor.Stride[0])

	slice = New(shape...)
	slice.Data = tensor.Data[index*firstStride : (index+1)*firstStride]

	return
}

func (tensor *Tensor) AssignSlice(index int, source Tensor) {
	destination := tensor.Slice(index)
	copy(destination.Data, source.Data)
}

func (tensor *Tensor) AssignReal(source Tensor) {
	tensor.AssignSlice(0, source)
}

func (tensor *Tensor) AssignDual(source Tensor) {
	tensor.AssignSlice(1, source)
}

func (tensor *Tensor) DualCopy() (dual Tensor) {
	dual = New(tensor.DualShape()...)
	copy(dual.Data, tensor.Data)

	return
}

func (tensor *Tensor) DualShape() (shape []int32) {
	shape = []int32{2}
	shape = append(shape, tensor.Shape...)
	return
}

func (tensor *Tensor) Real() (real Tensor) {
	real = tensor.Slice(0)
	return
}

func (tensor *Tensor) Dual() (dual Tensor) {
	dual = tensor.Slice(1)
	return
}
