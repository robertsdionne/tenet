package ten

import (
	"math"
)

func RectifiedLinear(x Tensor) (y Tensor) {
	y = NewLike(x)

	for i := range y.Data {
		y.Data[i] = rectifiedLinear(x.Data[i])
	}

	return
}

func rectifiedLinear(x float64) (y float64) {
	y = math.Max(0, x)
	return
}

func DualRectifiedLinear(x Tensor) (y Tensor) {
	y = NewLike(x)

	y.AssignReal(RectifiedLinear(x.Real()))

	for i := range y.Real().Data {
		if x.Real().Data[i] > 0 {
			y.Dual().Data[i] = x.Dual().Data[i]
		}
	}

	return
}

func RectifiedLinearGradient(dy, x Tensor) (dx Tensor) {
	dx = NewLike(x)

	for i := range x.Data {
		dx.Data[i] = rectifiedLinearGradient(dy.Data[i], x.Data[i])
	}

	return
}

func rectifiedLinearGradient(dy, x float64) (dx float64) {
	if x > 0 {
		dx = dy
	}
	return
}
