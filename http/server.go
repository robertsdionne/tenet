package http

import (
	"flag"
	"fmt"
	"net/http"
)

var port = flag.Int("http-port", 8080, "The HTTP port.")

// ListenAndServe starts the HTTP server.
func ListenAndServe() (err error) {
	err = http.ListenAndServe(fmt.Sprintf(":%d", *port), nil)
	return
}
