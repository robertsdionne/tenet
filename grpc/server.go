package grpc

import (
	"flag"
	"fmt"
	"github.com/golang/protobuf/ptypes/empty"
	"github.com/robertsdionne/tenet/prot"
	"github.com/robertsdionne/tenet/ten"
	"golang.org/x/net/context"
	"google.golang.org/grpc"
	"log"
	"net"
)

var port = flag.Int("grpc-port", 8080, "The GRPC port.")

// ListenAndServe starts the GRPC server.
func ListenAndServe() (err error) {
	listener, err := net.Listen("tcp", fmt.Sprintf(":%d", *port))
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}
	server := grpc.NewServer()
	prot.RegisterTeNetServer(server, newServer())
	err = server.Serve(listener)
	return
}

type server struct {
	W0, b0       ten.Tensor
	W10, W11, b1 ten.Tensor
}

const (
	width = 28
	size  = width * width
)

func newServer() (serve *server) {
	serve = &server{
		W0:  ten.Normal(0, 0.01)(size, size),
		b0:  ten.Constant(0.01)(size, 1),
		W10: ten.Normal(0, 0.01)(size, size),
		W11: ten.Normal(0, 0.01)(size, 10),
		b1:  ten.Constant(0.01)(size, 1),
	}
	return
}

func (serve *server) Get(ctx context.Context, in *empty.Empty) (response *prot.GetResponse, err error) {
	response = &prot.GetResponse{
		Shape: map[string]*prot.Shape{
			"x": {
				Components: []int32{size, 1},
			},
			"y": {
				Components: []int32{10, 1},
			},
		},
	}
	return
}

func (serve *server) Post(stream prot.TeNet_PostServer) (err error) {
	for {
		request, err := stream.Recv()
		if err != nil {
			return err
		}

		log.Println(request)

		x := ten.Tensor(*request.Tensors["x"])
		label := ten.Tensor(*request.Tensors["y"])
		log.Println(x)

		y, dx := serve.process(x, label)
		log.Println(y)

		dx_ := prot.Tensor(dx)

		err = stream.Send(&prot.PostResponse{
			Gradients: map[string]*prot.Tensor{
				"x": &dx_,
			},
		})
		if err != nil {
			return err
		}
	}
}

const (
	learning_rate = 0.001
)

func (serve *server) process(x, label ten.Tensor) (y, dx ten.Tensor) {
	h0 := ten.MatrixMultiply(serve.W0, x)
	h1 := ten.BroadcastAdd(h0, serve.b0)
	h2 := ten.RectifiedLinear(h1)
	y = ten.Add(h2, x)

	g0 := ten.MatrixMultiply(serve.W10, y)
	g1 := ten.MatrixMultiply(serve.W11, label)
	g2 := ten.Add(g0, g1)
	g3 := ten.BroadcastAdd(g2, serve.b1)
	g4 := ten.RectifiedLinear(g3)
	dy := ten.Add(g4, y)

	dh2, dx0 := ten.AddGradient(dy)
	dh1 := ten.RectifiedLinearGradient(dh2, h1)
	dh0, db0 := ten.BroadcastAddGradient(dh1, serve.b0)
	dw0, dx1 := ten.MatrixMultiplyGradient(dh0, serve.W0, x)
	dx = ten.Add(dx0, dx1)

	for i := range dw0.Data {
		serve.W0.Data[i] -= learning_rate * dw0.Data[i]
	}
	for i := range db0.Data {
		serve.b0.Data[i] -= learning_rate * db0.Data[i]
	}

	return
}
