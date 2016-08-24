package ten

import (
	"github.com/robertsdionne/tenet/prot"
)

// Shape is the shape of a Tensor.
type Shape []int32

// ShapeMap describes the shapes of a set of tensors.
type ShapeMap map[string]Shape

// Tensor is a multidimensional array of floating point values.
type Tensor prot.Tensor

// TensorMap describes the values of a set of tensors.
type TensorMap map[string]Tensor
