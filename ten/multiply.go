package ten

// Multiply multiplies two tensors.
func Multiply(a, b Tensor) (c Tensor) {
	c = NewLike(a)

	for i := range c.Data {
		c.Data[i] = a.Data[i] * b.Data[i]
	}

	return
}

func DualMultiply(a, b Tensor) (c Tensor) {
	c = NewLike(a)

	for i := range c.Real().Data {
		c.Real().Data[i] = a.Real().Data[i] * b.Real().Data[i]
		c.Dual().Data[i] = a.Real().Data[i]*b.Dual().Data[i] + a.Dual().Data[i]*b.Real().Data[i]
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
