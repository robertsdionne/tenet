package mnist

import (
	"github.com/robertsdionne/tenet/mod"
	"github.com/robertsdionne/tenet/ten"
	"log"
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

func (model *loss) Train(tensors ten.TensorMap, callback mod.Callback) (gradients ten.TensorMap, done chan bool) {
	x := tensors["x"]
	label := tensors["label"]

	loss, dx := ten.CrossEntropy(label, x)

	log.Println("Loss:", loss)

	gradients = ten.TensorMap{
		"x": dx,
	}

	done <- true

	return
}
