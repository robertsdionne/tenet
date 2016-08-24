package grpc

import (
	"flag"
	"fmt"
	"github.com/golang/protobuf/ptypes/empty"
	"github.com/robertsdionne/tenet/prot"
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

type server struct{}

func newServer() (serve *server) {
	serve = &server{}
	return
}

func (serve *server) Get(ctx context.Context, in *empty.Empty) (response *prot.GetResponse, err error) {
	response = &prot.GetResponse{
		Shape: map[string]*prot.Shape{
			"x": {
				Components: []int32{2, 2, 2},
			},
			"y": {
				Components: []int32{2, 2, 2},
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
		err = stream.Send(&prot.PostResponse{
			Gradients: request.Tensors,
		})
		if err != nil {
			return err
		}
	}
}
