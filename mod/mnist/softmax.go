package mnist

import (
	"github.com/robertsdionne/tenet/mod"
	"github.com/robertsdionne/tenet/ten"
	"log"
)

type softmax struct {
	syntheticGradient
	W0, b0      ten.Tensor
	averageLoss float64
}

func NewSoftmax(dimension, classes int32) (model mod.Model) {
	model = &softmax{
		W0: ten.Normal(μ, σ)(classes, dimension),
		b0: ten.Constant(β)(classes, 1),
		syntheticGradient: syntheticGradient{
			W10: ten.Constant(0)(classes, classes),
			W11: ten.Constant(0)(classes, classes),
			b1:  ten.Constant(0)(classes, 1),
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

func softmaxForward(W, b, x ten.Tensor) (h0, h1, y ten.Tensor) {
	h0 = ten.MatrixMultiply(W, x)
	h1 = ten.BroadcastAdd(h0, b)
	y = ten.Logistic(h1)
	return
}

func dualSoftmaxForward(W, b, x ten.Tensor) (h0, h1, y ten.Tensor) {
	h0 = ten.DualMatrixMultiply(W, x)
	h1 = ten.DualBroadcastAdd(h0, b)
	y = ten.DualLogistic(h1)
	return
}

func softmaxGradient(W, b, x, y, dy ten.Tensor) (dw, db, dx, dh0, dh1 ten.Tensor) {
	dh1 = ten.LogisticGradient(dy, y)
	dh0, db = ten.BroadcastAddGradient(dh1, b)
	dw, dx = ten.MatrixMultiplyGradient(dh0, W, x)
	return
}

func (model *softmax) process(x, label ten.Tensor) (y, dy, dx ten.Tensor, backprop backpropagate) {
	_, _, y = softmaxForward(model.W0, model.b0, x)
	dy = syntheticGradientForward(model.W10, model.W11, model.b1, y, label)
	dw0, db0, dx, _, _ := softmaxGradient(model.W0, model.b0, x, y, dy)

	backprop = func(loss ten.Tensor) {
		dw10, dw11, db1 := syntheticGradientGradient(model.W10, model.W11, model.b1, y, label, loss)

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
			W10.Data[i] += λ * dw10.Data[i]
		}
		for i := range dw11.Data {
			W11.Data[i] += λ * dw11.Data[i]
		}
		for i := range db1.Data {
			b1.Data[i] += λ * db1.Data[i]
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

	model.averageLoss = α*loss + (1-α)*model.averageLoss

	log.Printf("Loss %.4g  Average %.4g", loss, model.averageLoss)

	backprop(d)
}
