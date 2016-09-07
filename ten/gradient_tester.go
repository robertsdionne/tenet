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

		for i := range duals[t].Slice(1).Data {
			duals[t].Slice(1).Data[i] = 1
			output := operation(duals...)

			gradients[tensors[t]].Data[i] = Sum(output.Slice(1))
			duals[t].Slice(1).Data[i] = 0
		}
	}

	return
}
