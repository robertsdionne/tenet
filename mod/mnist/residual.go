package mnist

import (
	"github.com/robertsdionne/tenet/mod"
	"github.com/robertsdionne/tenet/ten"
	"log"
)

type residual struct {
	syntheticGradient
	W0, b0      ten.Tensor
	averageLoss float64
}

const (
	β = 0.01
	λ = 0.001
	μ = 0.0
	σ = 0.01
)

func NewResidual(dimension, classes int32) (model mod.Model) {
	model = &residual{
		W0: ten.Normal(μ, σ)(dimension, dimension),
		b0: ten.Constant(β)(dimension, 1),
		syntheticGradient: syntheticGradient{
			W10: ten.Normal(μ, σ)(dimension, dimension),
			W11: ten.Normal(μ, σ)(dimension, classes),
			b1:  ten.Constant(β)(dimension, 1),
		},
	}
	return
}

func (model *residual) Inputs() (shapes ten.ShapeMap) {
	shapes = ten.ShapeMap{
		"x":     []int32{model.W0.Shape[0], 1},
		"label": []int32{model.W11.Shape[1], 1},
	}
	return
}

func (model *residual) Train(tensors ten.TensorMap, callback mod.Callback) (gradients ten.TensorMap) {
	x := tensors["x"]
	label := tensors["label"]

	y, dy, dx, backprop := model.process(x, label)

	go model.propagate(y, dy, label, callback, backprop)

	gradients = ten.TensorMap{
		"x": dx,
	}

	return
}

type backpropagate func(loss ten.Tensor)

func (model *residual) process(x, label ten.Tensor) (y, dy, dx ten.Tensor, backprop backpropagate) {
	h0 := ten.MatrixMultiply(model.W0, x)
	h1 := ten.BroadcastAdd(h0, model.b0)
	y = ten.RectifiedLinear(h1)

	g0 := ten.MatrixMultiply(model.W10, y)
	g1 := ten.MatrixMultiply(model.W11, label)
	g2 := ten.Add(g0, g1)
	dy = ten.BroadcastAdd(g2, model.b1)

	dh1 := ten.RectifiedLinearGradient(y, h1)
	dh0, db0 := ten.BroadcastAddGradient(dh1, model.b0)
	dw0, dx := ten.MatrixMultiplyGradient(dh0, model.W0, x)

	backprop = func(loss ten.Tensor) {
		dg2, db1 := ten.BroadcastAddGradient(loss, model.b1)
		dg0, dg1 := ten.AddGradient(dg2)
		dw11, _ := ten.MatrixMultiplyGradient(dg1, model.W11, label)
		dw10, _ := ten.MatrixMultiplyGradient(dg0, model.W10, y)

		W0 := model.W0.Copy()
		b0 := model.b0.Copy()

		W10 := model.W10.Copy()
		W11 := model.W11.Copy()
		b1 := model.b1.Copy()

		for i := range dw0.Data {
			W0.Data[i] -= λ * dw0.Data[i]
		}
		for i := range db0.Data {
			b0.Data[i] -= λ * db0.Data[i]
		}

		for i := range dw10.Data {
			W10.Data[i] -= λ * dw10.Data[i]
		}
		for i := range dw11.Data {
			W11.Data[i] -= λ * dw11.Data[i]
		}
		for i := range db1.Data {
			b1.Data[i] -= λ * db1.Data[i]
		}

		model.W0 = W0
		model.b0 = b0

		model.W10 = W10
		model.W11 = W11
		model.b1 = b1

		return
	}

	return
}

func (model *residual) propagate(y, dy, label ten.Tensor, callback mod.Callback, backprop backpropagate) {
	gradients := callback(ten.TensorMap{
		"x":     y,
		"label": label,
	})

	d := ten.Subtract(gradients["x"], dy)
	var loss float64
	for _, value := range d.Data {
		loss += value * value / 2.0
	}

	model.averageLoss = α*loss + (1-α)*model.averageLoss

	log.Printf("Loss %.4g  Average %.4g", loss, model.averageLoss)

	backprop(d)
}
