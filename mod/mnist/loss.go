package mnist

import (
	"github.com/robertsdionne/tenet/mod"
	"github.com/robertsdionne/tenet/ten"
	"log"
	"math"
)

type loss struct {
	classes int32
}

func NewLoss(classes int32) (model mod.Model) {
	model = &loss{
		classes: classes,
	}
	return
}

func (model *loss) Inputs() (shapes ten.ShapeMap) {
	shapes = ten.ShapeMap{
		"x":     []int32{model.classes, 1},
		"label": []int32{model.classes, 1},
	}
	return
}

func (model *loss) Train(tensors ten.TensorMap, callback mod.Callback) (gradients ten.TensorMap) {
	x := tensors["x"]
	label := tensors["label"]

	loss, dx := ten.CrossEntropy(label, x)

	xArgmax, _ := findArgmax(x)
	yArgmax, _ := findArgmax(label)

	status := "✅"
	if xArgmax != yArgmax {
		status = "❌"
	}

	log.Println(status, "Actual:", xArgmax, "Expected:", yArgmax, "Loss:", loss)

	gradients = ten.TensorMap{
		"x": dx,
	}

	if math.IsInf(loss, 0) || math.IsNaN(loss) {
		log.Println(loss, label.Data, x.Data, dx.Data)
		gradients["x"] = ten.NewLike(x)
	}

	return
}

func findArgmax(tensor ten.Tensor) (argmax int, maximum float64) {
	argmax = -1
	maximum = math.Inf(-1)
	for i, value := range tensor.Data {
		if maximum < value {
			argmax, maximum = i, value
		}
	}
	return
}
