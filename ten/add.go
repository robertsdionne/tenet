package ten

func Add(a, b Tensor) (c Tensor) {
	c = NewLike(a)

	for i := range c.Data {
		c.Data[i] = a.Data[i] + b.Data[i]
	}

	return
}

func AddScalar(a Tensor, b float64) (c Tensor) {
	c = NewLike(a)

	for i := range c.Data {
		c.Data[i] = a.Data[i] + b
	}

	return
}

func DualAdd(a, b Tensor) (c Tensor) {
	c = Add(a, b)
	return
}

func DualAddScalar(a Tensor, bReal, bDual float64) (c Tensor) {
	c = NewLike(a)

	c.AssignReal(AddScalar(a.Real(), bReal))
	c.AssignDual(AddScalar(a.Dual(), bDual))

	return
}

func AddGradient(dc Tensor) (da, db Tensor) {
	da, db = NewLike(dc), NewLike(dc)

	for i := range dc.Data {
		da.Data[i] = dc.Data[i]
		db.Data[i] = dc.Data[i]
	}

	return
}
