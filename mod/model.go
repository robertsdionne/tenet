package mod

import (
	"github.com/robertsdionne/tenet/ten"
)

type Callback func(ten.TensorMap) ten.TensorMap

// Model describes the interface to a model.
type Model interface {
	Inputs() ten.ShapeMap
	Train(ten.TensorMap, Callback) (ten.TensorMap, chan bool)
}
