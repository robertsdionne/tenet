package main

import (
	"flag"
	"github.com/robertsdionne/tenet/grpc"
	"log"
)

func main() {
	flag.Parse()
	log.Fatalln(grpc.ListenAndServe())
}
