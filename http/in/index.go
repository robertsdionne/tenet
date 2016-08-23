package in

import (
	"encoding/json"
	"github.com/robertsdionne/tenet/mod"
	"github.com/robertsdionne/tenet/ten"
	"net/http"
)

// Controller holds HTTP handlers.
type Controller struct {
	model mod.Model
}

// Setup sets up the handlers for /in.
func (controller Controller) Setup(model mod.Model) {
	controller.model = model

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

type getResponse map[string]ten.Shape

func (controller *Controller) get(w http.ResponseWriter, r *http.Request) {
	json.NewEncoder(w).Encode(controller.model.Inputs())
}

type postRequest map[string]ten.Tensor

type postResponse map[string]ten.Tensor

func (controller *Controller) post(w http.ResponseWriter, r *http.Request) {
	request := postRequest{}
	err := json.NewDecoder(r.Body).Decode(&request)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	json.NewEncoder(w).Encode(request)
}
