package main

import (
	"fmt"
	"github.com/robertsdionne/tenet/dat"
	"log"
)

func main() {
	trainImages, trainLabels, _, _, err := dat.GetMNISTDataset()
	if err != nil {
		log.Fatalln(err)
	}

	for k := 0; k < int(trainImages.Shape[0]); k++ {
		for i := 0; i < int(trainImages.Shape[1]); i++ {
			for j := 0; j < int(trainImages.Shape[2]); j++ {
				value := *trainImages.At(k, i, j)
				switch {
				case value < 51:
					fmt.Print("  ")
				case value < 102:
					fmt.Print("░░")
				case value < 153:
					fmt.Print("▒▒")
				case value < 204:
					fmt.Print("▓▓")
				default:
					fmt.Print("██")
				}
			}
			fmt.Println()
		}
		fmt.Println(*trainLabels.At(k))
	}
}
