package main

import (
	"flag"
	"github.com/robertsdionne/tenet/grpc"
	"github.com/robertsdionne/tenet/http"
	"github.com/robertsdionne/tenet/prot"
	"log"
)

var protocol = prot.GRPC

func main() {
	flag.Var(&protocol, "protocol", "The server's protocol.")
	flag.Parse()

	switch protocol {
	case prot.GRPC:
		log.Fatalln(grpc.ListenAndServe())
	case prot.HTTP:
		log.Fatalln(http.ListenAndServe())
	case prot.HTTP2:
		log.Fatalln("HTTP2 is not yet implemented.")
	default:
		log.Fatalln("Unknown protocol.")
	}
}
