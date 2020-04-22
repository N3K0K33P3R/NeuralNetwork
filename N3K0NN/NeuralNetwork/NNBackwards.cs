using System;

namespace N3K0NN.NeuralNetwork
{
    public class NNBackwards<T>
    {
        public           double      LearningRate { get; }
        public           double      Momentum     { get; set; }
        private readonly NN          _nn;
        private readonly IDataSet<T> _dataSet;

        public NNBackwards(NN nn, IDataSet<T> _dataSet, double learningRate, double momentum)
        {
            Momentum      = momentum;
            LearningRate  = learningRate;
            _nn           = nn;
            this._dataSet = _dataSet;
        }

        public void Learn(T data)
        {
            var expected = _dataSet.GetData(data);
            GetNeuronErrors(expected);
            CorrectWeights();
        }

        private void GetNeuronErrors(double[] expected)
        {
            for (int i = _nn.Layers.Length - 1; i >= 0; i--)
            {
                for (int j = 0; j < _nn.Layers[i].Length; j++)
                {
                    Layer layer = _nn.Layers[i];
                    if (i == _nn.Layers.Length - 1)
                    {
                        layer.Neurons[j].error = (expected[j] - layer.Outputs[j]) * layer.Neurons[j].GetDerivative();
                    }
                    else if (!(layer.Neurons[j] is BiasNeuron))
                    {
                        double error = 0;
                        for (int k = 0; k < layer.Next.Length; k++)
                        {
                            error += layer.Next.Neurons[k].error * layer.Next.Neurons[k].Weights[j];
                        }

                        layer.Neurons[j].error = error * layer.Neurons[j].GetDerivative();
                    }
                }
            }
        }

        private void CorrectWeights()
        {
            for (int i = _nn.Layers.Length - 1; i >= 0; i--)
            {
                var layer = _nn.Layers[i];
                for (int j = 0; j < layer.Length; j++)
                {
                    var neuron = layer.Neurons[j];
                    for (int k = 0; k < neuron.Weights.Length; k++)
                    {
                        var prevNeuron = i == 0 ? _nn.Input.Neurons[k] : _nn.Layers[i - 1].Neurons[k];
                        var gradient   = prevNeuron.output * neuron.error;
                        neuron.DeltaWeights[k] = LearningRate * gradient + Momentum * neuron.DeltaWeights[k];

                        neuron.Weights[k] += neuron.DeltaWeights[k];
                    }
                }
            }
        }
    }
}