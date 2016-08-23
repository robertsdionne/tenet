package main

import (
	"flag"
	"github.com/robertsdionne/tenet/prot"
	"log"
)

var protocol = prot.HTTP

func main() {
	flag.Var(&protocol, "protocol", "The server's protocol.")
	flag.Parse()
	log.Println(protocol)
}
