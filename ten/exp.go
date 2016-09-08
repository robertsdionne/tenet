package ten

import (
	"math"
)

func Exp(x Tensor) (y Tensor) {
	y = NewLike(x)

	for i := range y.Data {
		y.Data[i] = math.Exp(x.Data[i])
	}

	return
}

func DualExp(x Tensor) (y Tensor) {
	y = Exp(x)
	return
}
