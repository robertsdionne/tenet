package ten

// Shape is the shape of a Tensor.
type Shape []int

// ShapeMap describes the shapes of a set of tensors.
type ShapeMap map[string]Shape

// Tensor is a multidimensional array of floating point values.
type Tensor struct {
	Data          []float64
	Shape, Stride Shape
}

// TensorMap describes the values of a set of tensors.
type TensorMap map[string]Tensor
