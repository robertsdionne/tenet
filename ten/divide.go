package ten

// Multiply multiplies two tensors.
func Divide(a, b Tensor) (c Tensor) {
	c = NewLike(a)

	for i := range a.Data {
		c.Data[i] = a.Data[i] / b.Data[i]
	}

	return
}

func DivideScalar(a Tensor, b float64) (c Tensor) {
	c = NewLike(a)

	for i := range a.Data {
		c.Data[i] = a.Data[i] / b
	}

	return
}

func DualDivideScalar(a Tensor, bReal, bDual float64) (c Tensor) {
	c = NewLike(a)

	c.AssignReal(DivideScalar(a.Real(), bReal))

	for i := range c.Dual().Data {
		c.Dual().Data[i] = (a.Dual().Data[i]*bReal - a.Real().Data[i]*bDual) / bReal / bReal
	}

	return
}
