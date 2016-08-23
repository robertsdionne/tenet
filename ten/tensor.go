package ten

// Shape is the shape of a Tensor.
type Shape []int

// Tensor is a multidimensional array of floating point values.
type Tensor struct {
	Data          []float64
	Shape, Stride Shape
}
