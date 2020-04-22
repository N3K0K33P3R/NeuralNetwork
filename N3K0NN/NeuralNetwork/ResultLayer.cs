﻿ namespace N3K0NN.NeuralNetwork
{
    internal class ResultLayer : Layer
    {
        internal double[] Result;

        internal ResultLayer() : base(0, 0, false, null)
        {
        }
        
        

        internal override void Activate(double[] inputs)
        {
            Result = inputs;
        }
    }
}