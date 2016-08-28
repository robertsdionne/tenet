package main

import (
	"flag"
	"fmt"
	"github.com/golang/protobuf/ptypes/empty"
	"github.com/robertsdionne/tenet/prot"
	"github.com/robertsdionne/tenet/ten"
	"golang.org/x/net/context"
	"google.golang.org/grpc"
	"log"
	"time"
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

	shape := response.Shape["x"].Components
	log.Printf("Response: %v", response.Shape["x"].Components)

	initializer := ten.Normal(0, 0.5)

	stream, err := client.Post(context.Background())
	if err != nil {
		log.Fatalf("failed to post: %v", err)
	}
	for {
		x := prot.Tensor(initializer(shape...))
		y := ten.New(10, 1)
		*y.At(5, 0) = 1
		label := prot.Tensor(y)
		log.Println("x", x)
		log.Println("y", y)
		err := stream.Send(&prot.PostRequest{
			Tensors: map[string]*prot.Tensor{
				"x":     &x,
				"label": &label,
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

		time.Sleep(5 * time.Second)
	}
}
