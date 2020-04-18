package perceptron

// Error represents project custom errors
type Error string

func (e Error) Error() string { return string(e) }
