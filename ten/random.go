package ten

import (
	"math/rand"
)

func Uniform(low, high float64) (initializer func(shape ...int32) Tensor) {
	initializer = func(shape ...int32) (tensor Tensor) {
		tensor = New(shape...)
		for i := range tensor.Data {
			tensor.Data[i] = low + (high-low)*rand.Float64()
		}
		return
	}
	return
}

func Normal(mean, standard_deviation float64) (initializer func(shape ...int32) Tensor) {
	initializer = func(shape ...int32) (tensor Tensor) {
		tensor = New(shape...)
		for i := range tensor.Data {
			tensor.Data[i] = rand.NormFloat64()*standard_deviation + mean
		}
		return
	}
	return
}
