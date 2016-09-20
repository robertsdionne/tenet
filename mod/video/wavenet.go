package main

import (
	"github.com/robertsdionne/tenet/ten"
	"log"
)

func WaveNet(wf, wg, wh []ten.Tensor, w0, w1, x ten.Tensor) (dwf, dwg, dwh []ten.Tensor, dw0, dw1, y ten.Tensor) {
	const dilation = 2

	af := make([]ten.Tensor, len(wf))
	ag := make([]ten.Tensor, len(wf))
	hf := make([]ten.Tensor, len(wf))
	hg := make([]ten.Tensor, len(wf))
	h0 := make([]ten.Tensor, len(wf))
	h1 := make([]ten.Tensor, len(wf))
	h := make([]ten.Tensor, len(wf)+1)
	sum := make([]ten.Tensor, len(wf))

	h[0] = x

	for i := 0; i < len(wf); i++ {
		log.Println("f", i)
		af[i] = ten.DilatedConvolutionVideo(wf[i], h[i], dilation)
		ag[i] = ten.DilatedConvolutionVideo(wg[i], h[i], dilation)
		hf[i] = ten.HyperbolicTangent(af[i])
		hg[i] = ten.Logistic(ag[i])
		h0[i] = ten.Multiply(hf[i], hg[i])
		h1[i] = ten.DilatedConvolutionVideo(wh[i], h0[i], 1)
		h[i+1] = ten.Add(h1[i], h[i])
	}

	for i := 0; i < len(wf); i++ {
		if i == 0 {
			sum[i] = h1[i]
		} else {
			sum[i] = ten.Add(sum[i-1], h1[i])
		}
	}

	hsum0 := ten.RectifiedLinear(sum[len(sum)-1])
	hsum1 := ten.DilatedConvolutionVideo(w0, hsum0, 1)
	hsum2 := ten.RectifiedLinear(hsum1)
	hsum3 := ten.DilatedConvolutionVideo(w1, hsum2, 1)
	y = ten.Logistic(hsum3)

	dy := ten.Constant(1)(y.Shape...)

	dsum := make([]ten.Tensor, len(wf))
	dh := make([]ten.Tensor, len(wf))
	dh1 := make([]ten.Tensor, len(wf))
	dh0 := make([]ten.Tensor, len(wf))
	dhg := make([]ten.Tensor, len(wf))
	dhf := make([]ten.Tensor, len(wf))
	dag := make([]ten.Tensor, len(wf))
	daf := make([]ten.Tensor, len(wf))
	dwh = make([]ten.Tensor, len(wf))
	dwg = make([]ten.Tensor, len(wf))
	dwf = make([]ten.Tensor, len(wf))

	dhsum3 := ten.LogisticGradient(dy, y)
	dw1, dhsum2 := ten.DilatedConvolutionVideoGradient(w1, hsum2, dhsum3, 1)
	dhsum1 := ten.RectifiedLinearGradient(dhsum2, hsum1)
	dw0, dhsum0 := ten.DilatedConvolutionVideoGradient(w0, hsum0, dhsum1, 1)

	dsum[len(dsum)-1] = ten.RectifiedLinearGradient(dhsum0, sum[len(sum)-1])

	for i := len(wf) - 1; i >= 0; i-- {
		if i == 0 {
			dh1[i] = dsum[i]
		} else {
			dsum[i-1], dh1[i] = ten.AddGradient(dsum[i])
		}
	}

	for i := len(wf) - 1; i >= 0; i-- {
		log.Println("b", i)
		if i < len(wf)-1 {
			var dh1i ten.Tensor
			dh1i, dh[i] = ten.AddGradient(dh[i+1])
			dh1[i] = ten.Add(dh1[i], dh1i)
		} else {
			dh[i] = ten.NewLike(h[i])
		}

		dwh[i], dh0[i] = ten.DilatedConvolutionVideoGradient(wh[i], h0[i], dh1[i], 1)
		dhf[i], dhg[i] = ten.MultiplyGradient(dh0[i], hf[i], hg[i])
		dag[i] = ten.LogisticGradient(dhg[i], hg[i])
		daf[i] = ten.HyperbolicTangentGradient(dhf[i], af[i])
		var dhi ten.Tensor
		dwg[i], dhi = ten.DilatedConvolutionVideoGradient(wg[i], h[i], dag[i], dilation)
		dh[i] = ten.Add(dh[i], dhi)
		dwf[i], dhi = ten.DilatedConvolutionVideoGradient(wf[i], h[i], daf[i], dilation)
		dh[i] = ten.Add(dh[i], dhi)
	}

	return
}

func WaveNetGradient(wf, wg, wh []ten.Tensor, w0, w1, x, y, dy ten.Tensor) (dx ten.Tensor) {
	return
}

func main() {
	wf := make([]ten.Tensor, 4)
	wg := make([]ten.Tensor, 4)
	wh := make([]ten.Tensor, 4)

	for i := range wf {
		wf[i] = ten.Normal(0, 1e-2)(2, 3, 3, 16, 16)
		wg[i] = ten.Normal(0, 1e-2)(2, 3, 3, 16, 16)
		wh[i] = ten.Normal(0, 1e-2)(1, 1, 1, 16, 16)
	}

	w0 := ten.Normal(0, 1e-2)(1, 1, 1, 16, 16)
	w1 := ten.Normal(0, 1e-2)(1, 1, 1, 16, 16)

	x := ten.Normal(0, 1)(16, 32, 20, 16)

	dwf, dwg, dwh, dw0, dw1, y := WaveNet(wf, wg, wh, w0, w1, x)

	log.Println(dwf, dwg, dwh, dw0, dw1, y)
}
