package http

import (
	"flag"
	"fmt"
	"github.com/robertsdionne/tenet/http/in"
	"net/http"
)

var port = flag.Int("http-port", 8080, "The HTTP port.")

// ListenAndServe starts the HTTP server.
func ListenAndServe() (err error) {
	server{}.setup()
	err = http.ListenAndServe(fmt.Sprintf(":%d", *port), nil)
	return
}

type server struct{}

func (s server) setup() {
	in.Controller{}.Setup()
}
