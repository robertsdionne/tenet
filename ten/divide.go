package ten

// Multiply multiplies two tensors.
func Divide(a, b Tensor) (c Tensor) {
	c = NewLike(a)

	for i := range a.Data {
		c.Data[i] = a.Data[i] / (b.Data[i] + ε)
	}

	return
}
