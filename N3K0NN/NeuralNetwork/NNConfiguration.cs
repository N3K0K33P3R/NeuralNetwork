using System;

namespace N3K0NN.NeuralNetwork
{
    public enum ActivateFunc
    {
        LeakyRelu,
        Logistic
    }

    [Serializable]
    public struct NnConfiguration
    {
        public bool         Bias            { get; set; }
        public int          LayersCount     { get; set; }
        public int[]        NeuronsInLayers { get; set; }
        public double[]     Code            { get; set; }
        public ActivateFunc ActivateFunc    { get; set; }
    }
}