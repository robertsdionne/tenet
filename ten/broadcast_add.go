package ten

func BroadcastAdd(x, b Tensor) (y Tensor) {
	y = NewLike(x)

	for i := 0; i < int(y.Shape[0]); i++ {
		for j := 0; j < int(y.Shape[1]); j++ {
			*y.At(i, j) = *x.At(i, j) + *b.At(i, 0)
		}
	}

	return
}

func DualBroadcastAdd(x, b Tensor) (y Tensor) {
	y = NewLike(x)

	y.AssignReal(BroadcastAdd(x.Real(), b.Real()))
	y.AssignDual(BroadcastAdd(x.Dual(), b.Dual()))

	return
}

func BroadcastAddGradient(dy, b Tensor) (dx, db Tensor) {
	dx, db = NewLike(dy), NewLike(b)

	for i := 0; i < int(dx.Shape[0]); i++ {
		for j := 0; j < int(dx.Shape[1]); j++ {
			*dx.At(i, j) = *dy.At(i, j)
		}
	}

	for i := 0; i < int(dx.Shape[0]); i++ {
		for j := 0; j < int(dx.Shape[1]); j++ {
			*db.At(i, 0) += *dy.At(i, j)
		}
	}

	return
}
