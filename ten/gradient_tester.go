package ten

func TestGradients(operation func(tensors ...Tensor) Tensor, tensors ...*Tensor) (gradients map[*Tensor]Tensor) {
	gradients = map[*Tensor]Tensor{}

	duals := []Tensor{}
	for _, tensor := range tensors {
		dual := tensor.DualCopy()
		duals = append(duals, dual)
	}

	for t := range tensors {
		gradients[tensors[t]] = NewLike(*tensors[t])

		for i := range duals[t].Dual().Data {
			duals[t].Dual().Data[i] = 1
			output := operation(duals...)

			gradients[tensors[t]].Data[i] = Sum(output.Dual())
			duals[t].Dual().Data[i] = 0
		}
	}

	return
}
