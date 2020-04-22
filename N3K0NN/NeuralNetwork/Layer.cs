using System;
using System.Collections.Generic;

namespace N3K0NN.NeuralNetwork
{
    [Serializable]
    internal class Layer
    {
        protected readonly bool     Bias;
        public readonly    Neuron[] Neurons;
        internal readonly  double[] Outputs;
        internal           Layer    Next;

        internal Layer(int count, int prevCount, bool bias, Func<double, double> activateFunc)
        {
            Bias    = bias;
            Neurons = new Neuron[bias ? count + 1 : count];
            for (var i = 0; i < count; i++)
            {
                Neurons[i] = new Neuron(activateFunc);
                var deltaWeights = new double[bias ? prevCount + 1 : prevCount];
                Neurons[i].DeltaWeights = deltaWeights;

                var randomWeights =
                    new double[bias ? prevCount + 1 : prevCount];
                for (var j = 0; j < randomWeights.Length; j++) randomWeights[j] = Static.Random.NextDouble() * 2 - 1;

                Neurons[i].Weights = randomWeights;
            }

            if (bias) Neurons[^1] = new BiasNeuron();

            Outputs = new double[Neurons.Length];
        }

        protected Layer()
        {
        }

        internal int Length => Bias ? Neurons.Length - 1 : Neurons.Length;

        internal void SetNextLayer(Layer next)
        {
            Next = next;
        }

        internal virtual void Activate(double[] inputs)
        {
            for (var i = 0; i < Neurons.Length; i++) Outputs[i] = Neurons[i].Activate(inputs);

            Next.Activate(Outputs);
        }

        internal List<double> GetCode()
        {
            var result = new List<double>();
            foreach (var neuron in Neurons) result.AddRange(neuron.Weights);

            return result;
        }

        internal void SetCode(List<double> code)
        {
            foreach (var neuron in Neurons) neuron.SetCode(code);
        }
    }
}