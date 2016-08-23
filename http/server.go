package http

import (
	"flag"
	"fmt"
	"github.com/robertsdionne/tenet/http/in"
	"github.com/robertsdionne/tenet/mod"
	"net/http"
)

var port = flag.Int("http-port", 8080, "The HTTP port.")

// ListenAndServe starts the HTTP server.
func ListenAndServe() (err error) {
	server{}.setup(mod.New())
	err = http.ListenAndServe(fmt.Sprintf(":%d", *port), nil)
	return
}

type server struct{}

func (s server) setup(model mod.Model) {
	in.Controller{}.Setup(model)
}
