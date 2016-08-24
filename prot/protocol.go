package prot

import (
	"fmt"
	"sort"
	"strings"
)

//go:generate protoc protocol.proto --go_out=plugins=grpc:.

// Protocol identifies the protocol the server will use.
type Protocol string

const (
	// GRPC identifies the Google Remote-Procedure-Call protocol.
	GRPC Protocol = "grpc"

	// HTTP identifies the HyperText Transer Protocol version 1.
	HTTP Protocol = "http"

	// HTTP2 identifies the HyperText Transfer Protocol version 2.
	HTTP2 Protocol = "http2"
)

var (
	protocols = []string{
		string(GRPC),
		string(HTTP),
		string(HTTP2),
	}
)

// Get gets the value of the Protocol string.Value.
func (protocol *Protocol) Get() interface{} {
	return Protocol(*protocol)
}

// Set sets the value of the Protocol flag.Value.
func (protocol *Protocol) Set(value string) (err error) {
	lowerValue := strings.ToLower(value)
	index := sort.SearchStrings(protocols, lowerValue)
	if index < len(protocols) && protocols[index] == lowerValue {
		*protocol = Protocol(value)
	} else {
		err = fmt.Errorf("%s is not a valid Protocol", value)
	}
	return
}

// String encodes the value of the Protocol flag.Value.
func (protocol *Protocol) String() (value string) {
	value = string(*protocol)
	return
}
