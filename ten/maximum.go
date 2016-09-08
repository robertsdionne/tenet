package ten

import (
	"math"
)

func Maximum(x Tensor) (index int, maximum float64) {
	index = -1
	maximum = math.Inf(-1)
	for i := range x.Data {
		if maximum < x.Data[i] {
			index = i
			maximum = x.Data[i]
		}
	}
	return
}

func DualMaximum(x Tensor) (index int, maximumReal, maximumDual float64) {
	index, maximumReal = Maximum(x.Real())
	maximumDual = x.Dual().Data[index]
	return
}
