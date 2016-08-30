package grpc

import (
	"flag"
	"fmt"
	"github.com/golang/protobuf/ptypes/empty"
	"github.com/robertsdionne/tenet/mod"
	"github.com/robertsdionne/tenet/mod/mnist"
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
	prot.RegisterTeNetServer(server, newServer(mnist.NewSoftmax(size, 10)))
	err = server.Serve(listener)
	return
}

type server struct {
	model mod.Model
}

const (
	width = 28
	size  = width * width
)

func newServer(model mod.Model) (serve *server) {
	serve = &server{
		model: model,
	}
	return
}

func (serve *server) Get(ctx context.Context, in *empty.Empty) (response *prot.GetResponse, err error) {
	response = &prot.GetResponse{
		Shape: map[string]*prot.Shape{},
	}
	for key, shape := range serve.model.Inputs() {
		response.Shape[key] = &prot.Shape{
			Components: shape,
		}
	}
	return
}

func (serve *server) Post(stream prot.TeNet_PostServer) (err error) {
	for {
		request, err := stream.Recv()
		if err != nil {
			return err
		}

		tensors := ten.TensorMap{}
		for key, tensor := range request.Tensors {
			tensors[key] = ten.Tensor(*tensor)
		}

		gradients, done := serve.model.Train(tensors, func(tensors ten.TensorMap) ten.TensorMap {
			log.Println(tensors["x"])
			return ten.TensorMap{
				"x": ten.New(10, 1),
			}
		})

		response := &prot.PostResponse{
			Gradients: map[string]*prot.Tensor{},
		}
		for key, gradient := range gradients {
			tensor := prot.Tensor(gradient)
			response.Gradients[key] = &tensor
		}

		err = stream.Send(response)
		if err != nil {
			<-done
			return err
		}

		<-done
	}
}
