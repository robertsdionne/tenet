package dat

import (
	"compress/gzip"
	"encoding/binary"
	"io"
	"log"
	"net/http"
	"net/url"
	"path"
)

const (
	source      = "http://yann.lecun.com/exdb/mnist/"
	trainImages = "train-images-idx3-ubyte.gz"
	trainLabels = "train-labels-idx1-ubyte.gz"
	testImages  = "t10k-images-idx3-ubyte.gz"
	testLabels  = "t10k-labels-idx1-ubyte.gz"
)

func GetDataset() {
	SaveFile(trainImages)
	SaveFile(trainLabels)
	SaveFile(testImages)
	SaveFile(testLabels)
}

func Read(reader io.ReadCloser) {
	var magicNumber int32
	err := binary.Read(reader, binary.BigEndian, &magicNumber)
	if err != nil {
		log.Fatalln(err)
	}

	log.Println(magicNumber)
}

func SaveFile(filename string) io.ReadCloser {
	fileURL, err := url.Parse(source)
	if err != nil {
		log.Fatalln(err)
	}

	fileURL.Path = path.Join(fileURL.Path, filename)
	log.Println("Retrieving", fileURL)

	response, err := http.Get(fileURL.String())
	if err != nil {
		log.Fatalln(err)
	}
	defer response.Body.Close()

	log.Println("Response", response.Status)

	reader, err := gzip.NewReader(response.Body)
	if err != nil {
		log.Fatalln(err)
	}

	Read(reader)

	return reader
	//
	// file, err := os.Create(filename)
	// if err != nil {
	// 	log.Fatalln(err)
	// }
	// defer file.Close()
	//
	// _, err = io.Copy(file, reader)
	// if err != nil {
	// 	log.Fatalln(err)
	// }
	//
	// log.Println("Done.")
}
