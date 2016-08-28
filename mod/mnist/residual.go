package mnist

import (
	"github.com/robertsdionne/tenet/mod"
	"github.com/robertsdionne/tenet/ten"
)

type residual struct {
	W0, b0       ten.Tensor
	W10, W11, b1 ten.Tensor
}

const (
	β = 0.01
	λ = 0.001
	μ = 0.0
	σ = 0.01
)

func NewResidual(dimension, classes int32) (model mod.Model) {
	model = &residual{
		W0:  ten.Normal(μ, σ)(dimension, dimension),
		b0:  ten.Constant(β)(dimension, 1),
		W10: ten.Normal(μ, σ)(dimension, dimension),
		W11: ten.Normal(μ, σ)(dimension, classes),
		b1:  ten.Constant(β)(dimension, 1),
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

func (model *residual) Train(
	tensors ten.TensorMap, callback mod.Callback) (gradients ten.TensorMap) {

	x := tensors["x"]
	label := tensors["label"]

	y, dy, dx, backpropagate := model.process(x, label)

	go model.propagate(y, dy, label, callback, backpropagate)

	gradients = ten.TensorMap{
		"x": dx,
	}

	return
}

type backpropagate func(loss ten.Tensor) (dw10, dw11, db1 ten.Tensor)

func (model *residual) process(x, label ten.Tensor) (y, dy, dx ten.Tensor, backprop backpropagate) {
	h0 := ten.MatrixMultiply(model.W0, x)
	h1 := ten.BroadcastAdd(h0, model.b0)
	h2 := ten.RectifiedLinear(h1)
	y = ten.Add(h2, x)

	g0 := ten.MatrixMultiply(model.W10, y)
	g1 := ten.MatrixMultiply(model.W11, label)
	g2 := ten.Add(g0, g1)
	g3 := ten.BroadcastAdd(g2, model.b1)
	g4 := ten.HyperbolicTangent(g3)
	dy = ten.Add(g4, y)

	backprop = func(loss ten.Tensor) (dw10, dw11, db1 ten.Tensor) {
		dg4, _ := ten.AddGradient(loss)
		dg3 := ten.HyperbolicTangentGradient(dg4, g4)
		dg2, db1 := ten.BroadcastAddGradient(dg3, model.b1)
		dg0, dg1 := ten.AddGradient(dg2)
		dw11, _ = ten.MatrixMultiplyGradient(dg1, model.W11, label)
		dw10, _ = ten.MatrixMultiplyGradient(dg0, model.W10, y)
		return
	}

	dh2, dx0 := ten.AddGradient(dy)
	dh1 := ten.RectifiedLinearGradient(dh2, h1)
	dh0, db0 := ten.BroadcastAddGradient(dh1, model.b0)
	dw0, dx1 := ten.MatrixMultiplyGradient(dh0, model.W0, x)
	dx = ten.Add(dx0, dx1)

	for i := range dw0.Data {
		model.W0.Data[i] -= λ * dw0.Data[i]
	}
	for i := range db0.Data {
		model.b0.Data[i] -= λ * db0.Data[i]
	}

	return
}

func (model *residual) propagate(y, dy, label ten.Tensor, callback mod.Callback, backprop backpropagate) {
	gradients := callback(ten.TensorMap{
		"x":     y,
		"label": label,
	})

	dw10, dw11, db1 := backprop(ten.Subtract(gradients["x"], dy))

	for i := range dw10.Data {
		model.W10.Data[i] -= λ * dw10.Data[i]
	}
	for i := range dw11.Data {
		model.W11.Data[i] -= λ * dw11.Data[i]
	}
	for i := range db1.Data {
		model.b1.Data[i] -= λ * db1.Data[i]
	}
}
