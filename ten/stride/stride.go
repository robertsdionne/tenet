package stride

// C calculates.
func C(shape ...int32) (stride []int32) {
	stride = make([]int32, len(shape))
	var product int32 = 1
	for i := range shape {
		index := len(shape) - 1 - i
		stride[index] = product
		product *= shape[index]
	}
	return
}

// Fortran calculates.
func Fortran(shape ...int32) (stride []int32) {
	return
}
