package perceptron_test

import (
	"testing"

	"github.com/rmescandon/perceptron"
	check "gopkg.in/check.v1"
)

var testCtx *testing.T

func Test(t *testing.T) {
	testCtx = t
	check.TestingT(t)
}

type NetworkSuite struct{}

var _ = check.Suite(&NetworkSuite{})

func (s *NetworkSuite) TestSimplestNetwork(c *check.C) {
	// Train the NAND function
	train_data := []perceptron.Sample{
		perceptron.Sample{X: []uint{0, 0}, Z: 1},
		perceptron.Sample{X: []uint{0, 1}, Z: 1},
		perceptron.Sample{X: []uint{1, 0}, Z: 1},
		perceptron.Sample{X: []uint{1, 1}, Z: 0},
	}

	var thresshold float32 = 0.5
	fn := func(output float32) uint {
		if output-thresshold > 0 {
			return 1
		}
		return 0
	}

	var learningRate float32 = 0.1
	var nLayers uint = 1
	var nNeurons uint = 1
	var bias uint = 1

	// Use one layer of one neuron. The simplest case
	p := perceptron.New(nLayers, nNeurons, fn, learningRate, bias)
	p.SetVerbose(true)
	err := p.Train(train_data, perceptron.ZeroWeightInit)
	c.Assert(err, check.IsNil)

	// Process entries after the training
	output, err := p.Process([]uint{1, 0, 0})
	c.Assert(err, check.IsNil)
	c.Assert(output, check.Equals, uint(1))

	output, err = p.Process([]uint{1, 0, 1})
	c.Assert(err, check.IsNil)
	c.Assert(output, check.Equals, uint(1))

	output, err = p.Process([]uint{1, 1, 0})
	c.Assert(err, check.IsNil)
	c.Assert(output, check.Equals, uint(1))

	output, err = p.Process([]uint{1, 1, 1})
	c.Assert(err, check.IsNil)
	c.Assert(output, check.Equals, uint(0))
}
