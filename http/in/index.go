package in

import (
	"encoding/json"
	"net/http"
)

// Controller holds HTTP handlers.
type Controller struct{}

// Setup sets up the handlers for /in.
func (controller Controller) Setup() {
	http.HandleFunc("/in", controller.index)
}

func (controller *Controller) index(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		controller.get(w, r)
	case http.MethodPost:
		controller.post(w, r)
	default:
		w.WriteHeader(http.StatusMethodNotAllowed)
	}
}

type getResponse struct{}

type postResponse struct{}

func (controller *Controller) get(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(getResponse{})
}

func (controller *Controller) post(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(postResponse{})
}
