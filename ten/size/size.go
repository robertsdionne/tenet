package size

// Of calculates.
func Of(shape ...int32) (size int) {
	size = 1
	for _, s := range shape {
		size *= int(s)
	}
	return
}
