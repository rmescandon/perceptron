package perceptron

import (
	"errors"
	"fmt"
)

const (
	errNoSamples       = Error("Not enough defined training sames")
	errNoIterationData = Error("Not enough defined training data in the iteration")
)

// Perceptron is a perceptron kind of neural network.
type Perceptron struct {
	layers  []*layer
	verbose bool
	bias    uint
}

// Sample represents a single sample input and its expected output.
// It is used for training purposed
type Sample struct {
	X []uint
	Z uint
}

type layer struct {
	neurons []*neuron
}

// New returns a perceptron of n layers, with m neurons per layer and with certain activation function
//
// TODO asuming a single layer so far
func New(nLayers, neuronsPerLayer uint, fn ActivationFunc, learningRate float32, bias uint) *Perceptron {
	p := &Perceptron{
		layers: make([]*layer, nLayers, nLayers),
		bias:   bias,
	}

	for i := uint(0); i < nLayers; i++ {
		ly := &layer{
			neurons: make([]*neuron, neuronsPerLayer, neuronsPerLayer),
		}

		for j := uint(0); j < neuronsPerLayer; j++ {
			ly.neurons[j] = newNeuron(fn, learningRate)
		}

		p.layers[i] = ly
	}

	return p
}

// SetVerbose sets to true or false if intermediate output must be shown
func (p *Perceptron) SetVerbose(v bool) {
	p.verbose = v
}

// Train trains the perceptron until having stable weights
func (p *Perceptron) Train(trainingData []Sample, ws WeightInitStrategy) error {
	n := len(trainingData)
	if n == 0 {
		return errNoSamples
	}

	m := len(trainingData[0].X)
	if m == 0 {
		return errNoIterationData
	}
	// calculate the length of the input and update
	// the defined neurons in the perceptron
	for _, ly := range p.layers {
		for _, n := range ly.neurons {
			// x[0] is assigned to bias, the rest of them x[1]..x[m] to the samples entries
			n.x = make([]uint, m+1, m+1)
			n.initWeights(ws)
			n.verbose = p.verbose
		}
	}

	// we need n last results to be stable (no correction on weights)
	// to admit having the training completed
	howMuchStablesInARow := 0

	if p.verbose {
		p.dumpHeaders(m)
	}

	// Process training data
	i := 0
	for {
		// obtain current sample from the training data
		sample := trainingData[i%n]
		i = i + 1

		// Update every neuron in the perceptron with the input
		x := sample.X
		x = append([]uint{p.bias}, x...)
		z := sample.Z

		for _, ly := range p.layers {
			for _, n := range ly.neurons {
				n.x = x
				n.z = z

				// train the neuron
				if err := n.train(); err != nil {
					return err
				}

				// propagate back the error
				if err := n.updateWeights(); err != nil {
					return err
				}
			}
		}

		if p.stable() {
			howMuchStablesInARow++
			if howMuchStablesInARow > n {
				break
			}
		} else {
			howMuchStablesInARow = 0
		}
	}

	return nil
}

// Process returns an output from a received input based on a previous training
func (p *Perceptron) Process(input []uint) (uint, error) {
	for _, ly := range p.layers {
		for _, n := range ly.neurons {
			n.x = input
			return n.process()
		}
	}
	// TODO REDEFINE THIS
	return 0, errors.New("Could not process")
}

// ended when last m iterations did not update the neuron weights
func (p *Perceptron) stable() bool {
	for _, ly := range p.layers {
		for _, n := range ly.neurons {
			if n.d > 0 {
				return false
			}
		}
	}
	return true
}

func (p *Perceptron) dumpHeaders(m int) {
	fmt.Println("Training:")
	fmt.Printf(" b ")
	for i := 1; i <= m; i++ {
		fmt.Printf(" x%d ", i)
	}
	fmt.Printf(" z ")
	for i := 0; i < m+1; i++ {
		fmt.Printf(" w%d ", i)
	}
	fmt.Printf(" y ")
	fmt.Printf(" d ")
	for i := 0; i < m-1; i++ {
		fmt.Printf(" w%d' ", i)
	}
	fmt.Println()
}
