package ten

import (
	"math"
)

func Logistic(x Tensor) (y Tensor) {
	y = NewLike(x)

	for i := range x.Data {
		y.Data[i] = logistic(x.Data[i])
	}

	return
}

func logistic(x float64) (y float64) {
	y = 1 / (1 + math.Exp(-x))
	return
}

func LogisticGradient(dy, y Tensor) (dx Tensor) {
	dx = NewLike(y)

	for i := range y.Data {
		dx.Data[i] = logisticGradient(dy.Data[i], y.Data[i])
	}

	return
}

func logisticGradient(dy, y float64) (dx float64) {
	dx = dy * y * (1 - y)
	return
}