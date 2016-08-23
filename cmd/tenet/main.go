package main

import (
	"flag"
	"github.com/robertsdionne/tenet/http"
	"github.com/robertsdionne/tenet/prot"
	"log"
)

var protocol = prot.HTTP

func main() {
	flag.Var(&protocol, "protocol", "The server's protocol.")
	flag.Parse()

	switch protocol {
	case prot.HTTP:
		log.Fatalln(http.ListenAndServe())
	case prot.HTTP2:
		log.Fatalln("HTTP2 is not yet implemented.")
	case prot.GRPC:
		log.Fatalln("GRPC is not yet implemented.")
	default:
		log.Fatalln("Unknown protocol.")
	}
}
