package mnist

import (
	"github.com/robertsdionne/tenet/ten"
)

type syntheticGradient struct {
	W10, W11, b1 ten.Tensor
}

func syntheticGradientForward(W0, W1, b, y, label ten.Tensor) (dy ten.Tensor) {
	g0 := ten.MatrixMultiply(W0, y)
	g1 := ten.MatrixMultiply(W1, label)
	g2 := ten.Add(g0, g1)
	dy = ten.BroadcastAdd(g2, b)
	return
}

func dualSyntheticGradientForward(W0, W1, b, y, label ten.Tensor) (dy ten.Tensor) {
	g0 := ten.DualMatrixMultiply(W0, y)
	g1 := ten.DualMatrixMultiply(W1, label)
	g2 := ten.DualAdd(g0, g1)
	dy = ten.DualBroadcastAdd(g2, b)
	return
}

func syntheticGradientGradient(W0, W1, b, y, label, loss ten.Tensor) (dw0, dw1, db ten.Tensor) {
	dg2, db := ten.BroadcastAddGradient(loss, b)
	dg0, dg1 := ten.AddGradient(dg2)
	dw1, _ = ten.MatrixMultiplyGradient(dg1, W1, label)
	dw0, _ = ten.MatrixMultiplyGradient(dg0, W0, y)
	return
}
