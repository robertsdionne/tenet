package ten

// DilatedConvolutionVideo convolves a dilated filter w with video signal x along both the time and spatial axes.
// The convolution along the time axis is causal and has stride matching the filter-duration.
// * Filter w has shape (filter-duration, height, width, in-channels, out-channels).
// * Signal x has shape (duration, height, width, in-channels).
// * Output y has shape (duration, height, width, out-channels).
func DilatedConvolutionVideo(w, x Tensor, dilation int) (y Tensor) {
	y = New(x.Shape[0], x.Shape[1], x.Shape[2], w.Shape[4])

	for t1 := 0; t1 < int(y.Shape[0]); t1++ {
		for i1 := 0; i1 < int(y.Shape[1]); i1++ {
			for j1 := 0; j1 < int(y.Shape[2]); j1++ {
				for t := 0; t < int(w.Shape[0]); t++ {
					for i := 0; i < int(w.Shape[1]); i++ {
						for j := 0; j < int(w.Shape[2]); j++ {
							I := dilation * (i - int(w.Shape[1])/2)
							J := dilation * (j - int(w.Shape[2])/2)
							T := dilation * t
							t0 := t1 + T
							i0 := i1 + I
							j0 := j1 + J

							if t0 < 0 || t0 >= int(x.Shape[0]) ||
								i0 < 0 || i0 >= int(x.Shape[1]) ||
								j0 < 0 || j0 >= int(x.Shape[2]) {
								continue
							}

							for in := 0; in < int(w.Shape[3]); in++ {
								for out := 0; out < int(w.Shape[4]); out++ {
									W := *w.At(t, i, j, in, out)
									X := *x.At(t0, i0, j0, in)
									*y.At(t1, i1, j1, out) += W * X
								}
							}
						}
					}
				}
			}
		}
	}

	return
}

// DilatedConvolutionVideoGradient calculates the gradients with respect to parameters of DilatedConvolutionVideo.
func DilatedConvolutionVideoGradient(w, x, dy Tensor, dilation int) (dw, dx Tensor) {
	return
}
