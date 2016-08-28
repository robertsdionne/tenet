package ten

import (
	"math"
)

const (
	ε = math.SmallestNonzeroFloat64
)

func Softmax(x Tensor) (y Tensor) {
	y = NewLike(x)

	maximum := math.Inf(-1)
	for i := range x.Data {
		if maximum < x.Data[i] {
			maximum = x.Data[i]
		}
	}

	sum := ε
	for i := range x.Data {
		sum += math.Exp(x.Data[i] - maximum)
	}

	for i := range x.Data {
		y.Data[i] = (math.Exp(x.Data[i]-maximum) + ε) / sum
	}

	return
}

func SoftmaxGradient(dy, y Tensor) (dx Tensor) {
	dx = NewLike(y)

	for i := range y.Data {
		for j := range y.Data {
			δij := 0.0
			if i == j {
				δij = 1.0
			}
			dx.Data[i] += y.Data[j] * dy.Data[j] * (δij - y.Data[i])
		}
	}

	return
}
