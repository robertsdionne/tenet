package mnist

import (
	"github.com/robertsdionne/tenet/mod"
	"github.com/robertsdionne/tenet/ten"
	"log"
)

type softmax struct {
	syntheticGradient
	W0, b0 ten.Tensor
}

func NewSoftmax(dimension, classes int32) (model mod.Model) {
	model = &softmax{
		W0: ten.Normal(μ, σ)(classes, dimension),
		b0: ten.Constant(β)(classes, 1),
		syntheticGradient: syntheticGradient{
			W10: ten.Normal(μ, σ)(classes, classes),
			W11: ten.Normal(μ, σ)(classes, classes),
			b1:  ten.Constant(β)(classes, 1),
		},
	}
	return
}

func (model *softmax) Inputs() (shapes ten.ShapeMap) {
	shapes = ten.ShapeMap{
		"x":     []int32{model.W0.Shape[1], 1},
		"label": []int32{model.W11.Shape[1], 1},
	}
	return
}

func (model *softmax) Train(tensors ten.TensorMap, callback mod.Callback) (gradients ten.TensorMap) {
	x := tensors["x"]
	label := tensors["label"]

	y, dy, dx, backprop := model.process(x, label)

	go model.propagate(y, dy, label, callback, backprop)

	gradients = ten.TensorMap{
		"x": dx,
	}

	return
}

func (model *softmax) process(x, label ten.Tensor) (y, dy, dx ten.Tensor, backprop backpropagate) {
	h0 := ten.MatrixMultiply(model.W0, x)
	h1 := ten.BroadcastAdd(h0, model.b0)
	y = ten.Softmax(h1)

	g0 := ten.MatrixMultiply(model.W10, y)
	g1 := ten.MatrixMultiply(model.W11, label)
	g2 := ten.Add(g0, g1)
	g3 := ten.BroadcastAdd(g2, model.b1)
	g4 := ten.HyperbolicTangent(g3)
	dy = ten.Add(g4, y)

	dh1 := ten.SoftmaxGradient(dy, y)
	dh0, db0 := ten.BroadcastAddGradient(dh1, model.b0)
	dw0, dx := ten.MatrixMultiplyGradient(dh0, model.W0, x)

	backprop = func(loss ten.Tensor) {
		dg4, _ := ten.AddGradient(loss)
		dg3 := ten.HyperbolicTangentGradient(dg4, g4)
		dg2, db1 := ten.BroadcastAddGradient(dg3, model.b1)
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

func (model *softmax) propagate(y, dy, label ten.Tensor, callback mod.Callback, backprop backpropagate) {
	gradients := callback(ten.TensorMap{
		"x":     y,
		"label": label,
	})

	d := ten.Subtract(gradients["x"], dy)
	var loss float64
	for _, value := range d.Data {
		loss += value * value / 2.0
	}
	log.Println(loss)

	backprop(d)
}
