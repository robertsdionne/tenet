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

var inputPort = flag.Int("input-port", 8080, "The GRPC port.")
var outputPort = flag.Int("output-port", 0, "The output port.")
var modelType = flag.String("model-type", "residual", "Which model: residual, softmax, loss")

// ListenAndServe starts the GRPC server.
func ListenAndServe() (err error) {
	listener, err := net.Listen("tcp", fmt.Sprintf(":%d", *inputPort))
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}

	var client prot.TeNetClient
	if *outputPort > 0 {
		connection, err := grpc.Dial(fmt.Sprintf(":%d", *outputPort), grpc.WithInsecure())
		if err != nil {
			log.Fatalf("failed to dial: %v", err)
		}
		defer connection.Close()
		client = prot.NewTeNetClient(connection)
	}

	server := grpc.NewServer()

	var model mod.Model

	switch *modelType {
	case "residual":
		model = mnist.NewResidual(size, 10)
	case "softmax":
		model = mnist.NewSoftmax(size, 10)
	case "loss":
		model = mnist.NewLoss(10)
	default:
		log.Fatalln("Unknown model type", *modelType)
	}

	prot.RegisterTeNetServer(server, newServer(model, client))
	err = server.Serve(listener)
	return
}

type server struct {
	model  mod.Model
	client prot.TeNetClient
}

const (
	width = 28
	size  = width * width
)

func newServer(model mod.Model, client prot.TeNetClient) (serve *server) {
	serve = &server{
		model:  model,
		client: client,
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

		gradients := serve.model.Train(tensors, func(tensors ten.TensorMap) ten.TensorMap {
			stream, err := serve.client.Post(context.Background())
			if err != nil {
				log.Fatalln(err)
			}

			request := &prot.PostRequest{
				Tensors: map[string]*prot.Tensor{},
			}
			for key, tensor := range tensors {
				protTensor := prot.Tensor(tensor)
				request.Tensors[key] = &protTensor
			}

			err = stream.Send(request)
			if err != nil {
				log.Fatalln(err)
			}

			response, err := stream.Recv()
			if err != nil {
				log.Fatalln(err)
			}

			gradients := ten.TensorMap{}
			for key, gradient := range response.Gradients {
				tensor := ten.Tensor(*gradient)
				gradients[key] = tensor
			}

			return gradients
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
			return err
		}
	}
}
