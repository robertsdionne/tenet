package ten

// Multiply multiplies two tensors.
func Multiply(a, b Tensor) (c Tensor) {
	c = NewLike(a)

	for i := 0; i < int(a.Shape[0]); i++ {
		for j := 0; j < int(a.Shape[1]); j++ {
			*c.At(i, j) = *a.At(i, j) * *b.At(i, j)
		}
	}

	return
}

// MultiplyGradient calculates the gradient of Multiply with respect to a and b.
func MultiplyGradient(dc, a, b Tensor) (da, db Tensor) {
	da, db = NewLike(dc), NewLike(dc)

	for i := range dc.Data {
		da.Data[i] = dc.Data[i] * b.Data[i]
		db.Data[i] = dc.Data[i] * a.Data[i]
	}
	return
}
