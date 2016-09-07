package main

import (
	"flag"
	"fmt"
	"github.com/golang/protobuf/ptypes/empty"
	"github.com/robertsdionne/tenet/dat"
	"github.com/robertsdionne/tenet/prot"
	"github.com/robertsdionne/tenet/ten"
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

	shape := response.Shape["x"].Components

	trainImages, trainLabels, _, _, err := dat.GetMNISTDataset()
	if err != nil {
		log.Fatalln(err)
	}

	stream, err := client.Post(context.Background())
	if err != nil {
		log.Fatalf("failed to post: %v", err)
	}
	step := 0
	for {
		i := step % int(trainImages.Shape[0])
		xData := ten.New(shape...)
		for j := 0; j < 28; j++ {
			for k := 0; k < 28; k++ {
				*xData.At(j*k, 0) = *trainImages.At(i, j, k)
			}
		}
		x := prot.Tensor(xData)
		y := ten.New(10, 1)
		log.Println(*trainLabels.At(i))
		*y.At(int(*trainLabels.At(i)), 0) = 1
		step++
		label := prot.Tensor(y)
		err := stream.Send(&prot.PostRequest{
			Tensors: map[string]*prot.Tensor{
				"x":     &x,
				"label": &label,
			},
		})
		if err != nil {
			log.Fatalf("failed to send: %v", err)
		}

		_, err = stream.Recv()
		if err != nil {
			log.Fatalf("failed to recv: %v", err)
		}
	}
}
