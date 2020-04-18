package perceptron

import (
	"fmt"
	"math/rand"
	"sync"
)

const (
	errOutOfBounds = Error("Index out of bounds")
)

// WeightInitStrategy the strategy of how to init the neuron weights
type WeightInitStrategy uint

// How to init the weights
const (
	ZeroWeightInit WeightInitStrategy = iota
	RandomWeightInit
)

// ActivationFunc defines the signature for the activation function
//
// some examples
// identity: La función de activación es f(x)=x.
// logistic: La función de activación es la función sigmoide f(x)=1/(1+exp(-x)).
// tanh: La función de activación es la función tangente hiperbolico f(x)=tanh(x).
// relu: La función de activación es función rectificada de recta unitaria f(x)=max(0,x).
//
// See https://ernestocrespo13.wordpress.com/2018/02/13/funciones-de-activacion-para-un-perceptron/
type ActivationFunc = func(float32) uint

// neuron represents a single neuron in a perceptron
type neuron struct {
	// array of inputs
	x []uint
	// bias
	b uint
	// desired output
	z uint
	// array of weights
	w []float32
	// activation function
	fn ActivationFunc
	// learning rate
	r float32
	// correction on last iteration
	d float32
	// output
	y uint

	mux     sync.Mutex
	verbose bool
}

func newNeuron(fn ActivationFunc, learningRate float32) *neuron {
	return &neuron{
		fn: fn,
		r:  learningRate,
	}
}

func (n *neuron) initWeights(st WeightInitStrategy) {
	xlen := len(n.x)
	n.w = make([]float32, xlen, xlen)

	var fn func() float32

	switch st {
	case ZeroWeightInit:
		fn = func() float32 { return 0 }
	case RandomWeightInit:
		fn = func() float32 { return rand.Float32() }
	}

	for i := range n.w {
		n.w[i] = fn()
	}
}

func (n *neuron) sensor(i int) (float32, error) {
	if len(n.x) <= i {
		return 0, errOutOfBounds
	}

	if len(n.w) <= i {
		return 0, errOutOfBounds
	}

	return float32(n.x[i]) * n.w[i], nil
}

func (n *neuron) train() error {
	var err error
	n.y, err = n.process()
	if err != nil {
		return err
	}

	if n.verbose {
		n.dumpXs()
		n.dumpZ()
		n.dumpWeights()
		n.dumpY()
		n.dumpD()
	}
	return nil
}

func (n *neuron) error() int {
	return int(n.z - n.y)
}

func (n *neuron) correction() float32 {
	return n.r * float32(n.error())
}

func (n *neuron) newWeight(i int) (float32, error) {
	if len(n.x) <= i {
		return -1, errOutOfBounds
	}

	// new weigth = existing weight + Æ(xj * d)
	return n.w[i] + n.d*float32(n.x[i]), nil
}

// backpropagation
func (n *neuron) updateWeights() error {
	// update the correction
	n.d = n.correction()

	for i := range n.x {
		w, err := n.newWeight(i)
		if err != nil {
			return err
		}

		if n.w[i] != w {
			n.w[i] = w
		}
	}

	if n.verbose {
		n.dumpWeights()
		fmt.Println()
	}

	return nil
}

func (n *neuron) process() (uint, error) {
	var output float32

	for i := range n.x {
		ci, err := n.sensor(i)
		if err != nil {
			return 0, err
		}
		output += ci
	}

	return n.fn(output), nil
}

func (n *neuron) dumpXs() {
	for _, x := range n.x {
		fmt.Printf(" %d ", x)
	}
}

func (n *neuron) dumpZ() {
	fmt.Printf(" %d ", n.z)
}

func (n *neuron) dumpWeights() {
	for _, w := range n.w {
		fmt.Printf(" %.2f ", w)
	}
}

func (n *neuron) dumpD() {
	fmt.Printf(" %.2f ", n.d)
}

func (n *neuron) dumpY() {
	fmt.Printf(" %d ", n.y)
}
