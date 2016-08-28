package ten

func Sum(t Tensor) (sum float64) {
	for _, value := range t.Data {
		sum += value
	}
	return
}
