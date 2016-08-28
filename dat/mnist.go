package dat

import (
	"compress/gzip"
	"encoding/binary"
	"errors"
	"fmt"
	"github.com/robertsdionne/tenet/ten"
	"github.com/robertsdionne/tenet/ten/size"
	"github.com/robertsdionne/tenet/ten/stride"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"path"
)

const (
	sourceURLPrefix     = "http://yann.lecun.com/exdb/mnist/"
	trainImagesFilename = "train-images-idx3-ubyte.gz"
	trainLabelsFilename = "train-labels-idx1-ubyte.gz"
	testImagesFilename  = "t10k-images-idx3-ubyte.gz"
	testLabelsFilename  = "t10k-labels-idx1-ubyte.gz"
)

func GetMNISTDataset() (trainImages, trainLabels, testImages, testLabels ten.Tensor, err error) {
	for _, filename := range []string{
		trainImagesFilename,
		trainLabelsFilename,
		testImagesFilename,
		testLabelsFilename,
	} {
		err = maybeDownload(filename)
		if err != nil {
			return
		}
	}

	trainImages, err = readTensor(trainImagesFilename)
	if err != nil {
		return
	}

	trainLabels, err = readTensor(trainLabelsFilename)
	if err != nil {
		return
	}

	testLabels, err = readTensor(testImagesFilename)
	if err != nil {
		return
	}

	testLabels, err = readTensor(testLabelsFilename)
	if err != nil {
		return
	}

	return
}

func maybeDownload(filename string) (err error) {
	if _, err = os.Stat(path.Join("data", filename)); !os.IsNotExist(err) {
		log.Println(filename, "exists; skipping download.")
		return
	}

	fileURL, err := url.Parse(sourceURLPrefix)
	if err != nil {
		return
	}

	fileURL.Path = path.Join(fileURL.Path, filename)

	log.Println("Downloading", fileURL)

	response, err := http.Get(fileURL.String())
	if err != nil {
		return
	}
	defer response.Body.Close()

	log.Println("Response", response.Status)

	file, err := os.Create(path.Join("data", filename))
	if err != nil {
		return
	}
	defer file.Close()

	_, err = io.Copy(file, response.Body)
	if err != nil {
		return
	}

	log.Println("Done.")

	return
}

const (
	magicNumberLabels = 0x0801
	magicNumberImages = 0x0803
)

func readTensor(filename string) (tensor ten.Tensor, err error) {
	filename = path.Join("data", filename)

	file, err := os.Open(filename)
	if err != nil {
		return
	}

	reader, err := gzip.NewReader(file)
	if err != nil {
		return
	}

	var magicNumber int32
	err = binary.Read(reader, binary.BigEndian, &magicNumber)
	if err != nil {
		return
	}

	switch magicNumber {
	case magicNumberLabels:
		tensor.Shape = make([]int32, 1)
	case magicNumberImages:
		tensor.Shape = make([]int32, 3)
	default:
		err = errors.New("Unknown magic number.")
	}

	for i := range tensor.Shape {
		err = binary.Read(reader, binary.BigEndian, &tensor.Shape[i])
		if err != nil {
			return
		}
	}

	tensor.Stride = stride.C(tensor.Shape...)

	n := size.Of(tensor.Shape...)
	tensor.Data = make([]float64, n)

	for i := range tensor.Data {
		var value uint8

		err = binary.Read(reader, binary.BigEndian, &value)
		if err != nil {
			return
		}

		tensor.Data[i] = float64(value)

		if i%(n/80) == 0 {
			fmt.Print(".")
		}
	}

	fmt.Println()

	return
}
