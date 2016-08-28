package ten

func Constant(constant float64) (initializer func(shape ...int32) Tensor) {
	initializer = func(shape ...int32) (tensor Tensor) {
		tensor = New(shape...)
		for i := range tensor.Data {
			tensor.Data[i] = constant
		}
		return
	}
	return
}
