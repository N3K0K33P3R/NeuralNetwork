﻿using System;

 namespace N3K0NN.NeuralNetwork
{
    [Serializable]
    internal class InputLayer : Layer
    {
        private readonly double[] _resWithBias;
        internal InputLayer(int count, bool bias) : base(count,0,bias, null)
        {
            if (!bias) return;
            _resWithBias     = new double[count +1];
            _resWithBias[^1] = 1;
        }

        private InputLayer()
        {
            
        }

        internal override void Activate(double[] inputs)
        {
            for (int i = 0; i < inputs.Length; i++)
            {
                Neurons[i].output = inputs[i];
            }
            if (Bias)
            {
                for (int i = 0; i < inputs.Length; i++)
                {
                    _resWithBias[i] = inputs[i];
                }

                Next.Activate(_resWithBias);
            }
            else
            {
                Next.Activate(inputs);
            }
        }
    }
}