using System;
using System.Collections.Generic;
using System.Linq;

namespace N3K0NN.NeuralNetwork
{
    [Serializable]
    internal class Neuron
    {
        internal Func<double, double> activateFunc;
        internal double               error;
        internal double               output;

        internal Neuron()
        {
        }

        internal Neuron(Func<double, double> activateFunc)
        {
            this.activateFunc = activateFunc;
        }

        internal double[] Weights      { get; set; }
        internal double[] DeltaWeights { get; set; }
        private  double[] Input        { get; set; }

        internal virtual double Activate(double[] values)
        {
            Input = values;
            var sum = Weights.Select((t, i) => values[i] * t).Sum();

            output = activateFunc(sum);
            return output;
        }

        internal void SetCode(List<double> code)
        {
            for (var i = 0; i < Weights.Length; i++)
            {
                Weights[i] = code[0];
                code.RemoveAt(0);
            }
        }

        internal double GetDerivative()
        {
            return ActivateFuncs.Derivatives[activateFunc](output);
        }
    }
}