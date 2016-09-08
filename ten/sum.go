package ten

func Sum(t Tensor) (sum float64) {
	for _, value := range t.Data {
		sum += value
	}
	return
}

func DualSum(t Tensor) (sumReal, sumDual float64) {
	sumReal = Sum(t.Real())
	sumDual = Sum(t.Dual())
	return
}
