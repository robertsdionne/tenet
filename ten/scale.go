package ten

func Scale(x Tensor, scale float64) (y Tensor) {
	y = NewLike(x)

	for i := range x.Data {
		y.Data[i] = x.Data[i] * scale
	}
	return
}
