package ten

import (
	"math"
)

func HyperbolicTangent(x Tensor) (y Tensor) {
	y = NewLike(x)

	for i := range x.Data {
		y.Data[i] = math.Tanh(x.Data[i])
	}

	return
}

func DualHyperbolicTangent(x Tensor) (y Tensor) {
	y = NewLike(x)

	for i := range x.Real().Data {
		y.Real().Data[i] = math.Tanh(x.Real().Data[i])
		cosh := math.Cosh(x.Real().Data[i])
		y.Dual().Data[i] = x.Dual().Data[i] / cosh / cosh
	}

	return
}

func HyperbolicTangentGradient(dy, y Tensor) (dx Tensor) {
	dx = NewLike(y)

	for i := range y.Data {
		dx.Data[i] = hyperbolicTangentGradient(dy.Data[i], y.Data[i])
	}

	return
}

func hyperbolicTangentGradient(dy, y float64) (dx float64) {
	dx = dy * (1 - y*y)
	return
}
