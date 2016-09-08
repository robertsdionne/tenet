package ten

import (
	"math"
)

const (
	ε = math.SmallestNonzeroFloat64
)

func Softmax(x Tensor) (y Tensor) {
	y = NewLike(x)

	_, maximum := Maximum(x)
	shifted := SubtractScalar(x, maximum)
	exp := Exp(shifted)
	sum := ε + Sum(exp)
	y = DivideScalar(AddScalar(exp, ε), sum)

	return
}

func DualSoftmax(x Tensor) (y Tensor) {
	y = NewLike(x)

	_, maximumReal, maximumDual := DualMaximum(x)
	shifted := DualSubtractScalar(x, maximumReal, maximumDual)
	exp := DualExp(shifted)
	sumReal, sumDual := DualSum(exp)
	sumReal += ε
	y = DualDivideScalar(DualAddScalar(exp, ε, 0), sumReal, sumDual)

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
