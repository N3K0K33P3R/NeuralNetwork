namespace N3K0NN.NeuralNetwork
{
    internal class BiasNeuron : Neuron
    {
        internal BiasNeuron()
        {
            Weights = new double[0];
            output = 1;
        }

        internal override double Activate(double[] values)
        {
            return 1;
        }
    }
}