package ten

import (
	"math"
)

// Log calculates the natural logarithm function.
func Log(x Tensor) (y Tensor) {
	y = NewLike(x)

	for i := range x.Data {
		y.Data[i] = math.Log(x.Data[i])
	}

	return
}
