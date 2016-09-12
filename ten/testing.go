package ten

func NewTest(shape ...int32) (tensor Tensor) {
	tensor = New(shape...)
	for i := range tensor.Data {
		index := tensor.Index(i)
		sum := 0
		for _, j := range index {
			sum += j
		}
		*tensor.At(index...) = float64(sum)
	}
	return
}

func NewTestLike(t0 Tensor) (t1 Tensor) {
	t1 = NewTest(t0.Shape...)
	return
}
