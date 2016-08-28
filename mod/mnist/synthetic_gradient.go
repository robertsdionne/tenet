package mnist

import (
	"github.com/robertsdionne/tenet/ten"
)

type syntheticGradient struct {
	W10, W11, b1 ten.Tensor
}
