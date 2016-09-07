package mnist

import (
	"github.com/robertsdionne/tenet/mod"
	"github.com/robertsdionne/tenet/ten"
	"log"
	"math"
)

const (
	α = 0.1
)

type loss struct {
	classes     int32
	averageLoss float64
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

	model.averageLoss = α*loss + (1-α)*model.averageLoss

	xArgmax, _ := findArgmax(x)
	yArgmax, _ := findArgmax(label)

	status := "✅"
	if xArgmax != yArgmax {
		status = "❌"
	}

	log.Printf("%s  Actual %d  Expected %d  Loss %.4g  Average %.4g\n", status, xArgmax, yArgmax, loss, model.averageLoss)

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
