package mod

import (
	"github.com/robertsdionne/tenet/ten"
)

// Model describes the interface to a model.
type Model interface {
	Inputs() ten.ShapeMap
}

type defaultModel struct{}

func (model *defaultModel) Inputs() (inputs ten.ShapeMap) {
	inputs = ten.ShapeMap{
		"x": {2, 2, 2},
		"y": {4, 4, 4},
	}
	return
}

// New creates a new model.
func New() (model Model) {
	model = &defaultModel{}
	return
}
