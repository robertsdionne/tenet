package ten

func CrossEntropy(target, y Tensor) (cost float64, dy Tensor) {
	onesLike := Constant(1 + 1e-10)

	cost = Sum(Scale(Add(
		Multiply(target, Log(y)),
		Multiply(Subtract(onesLike(target.Shape...), target), Log(Subtract(onesLike(y.Shape...), y)))), -1))

	dy = Divide(Divide(Subtract(y, target), y), Subtract(onesLike(y.Shape...), y))

	return
}
