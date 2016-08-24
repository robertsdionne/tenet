package main

import (
	"flag"
	"fmt"
	"github.com/golang/protobuf/ptypes/empty"
	"github.com/robertsdionne/tenet/prot"
	"golang.org/x/net/context"
	"google.golang.org/grpc"
	"log"
)

var port = flag.Int("port", 8080, "The server port.")

func main() {
	flag.Parse()
	connection, err := grpc.Dial(fmt.Sprintf(":%d", *port), grpc.WithInsecure())
	if err != nil {
		log.Fatalf("failed to dial: %v", err)
	}
	defer connection.Close()

	client := prot.NewTeNetClient(connection)

	response, err := client.Get(context.Background(), &empty.Empty{})
	if err != nil {
		log.Fatalf("failed to get: %v", err)
	}

	log.Printf("Response: %v", response)

	stream, err := client.Post(context.Background())
	if err != nil {
		log.Fatalf("failed to post: %v", err)
	}
	for {
		err := stream.Send(&prot.PostRequest{
			Tensors: map[string]*prot.Tensor{
				"x": {
					Data:  []float64{0, 1, 2, 3, 4, 5, 6, 7},
					Shape: []int32{2, 2, 2},
				},
				"y": {
					Data:  []float64{7, 6, 5, 4, 3, 2, 1, 0},
					Shape: []int32{2, 2, 2},
				},
			},
		})
		if err != nil {
			log.Fatalf("failed to send: %v", err)
		}

		response, err := stream.Recv()
		if err != nil {
			log.Fatalf("failed to recv: %v", err)
		}

		log.Printf("Response: %v", response)
	}
}
