using System;
using System.Collections.Generic;
using System.Linq;

namespace N3K0NN.NeuralNetwork
{
    [Serializable]
    public class NN
    {
        internal readonly InputLayer      Input;
        internal readonly Layer[]         Layers;
        private           NnConfiguration _nnConfiguration;
        private readonly  ResultLayer     _result;


        public NN(NnConfiguration configuration)
        {
            Layers = new Layer[configuration.LayersCount - 1];

            var activateFunc = configuration.ActivateFunc switch
            {
                ActivateFunc.Logistic  => (Func<double, double>) ActivateFuncs.LogisticFunc,
                ActivateFunc.LeakyRelu => ActivateFuncs.LeakyReLu,
                _                      => null
            };

            Layer prev = Input = new InputLayer(configuration.NeuronsInLayers[0], configuration.Bias);
            for (var i = 1; i < configuration.LayersCount; i++)
            {
                Layers[i - 1] = new Layer(configuration.NeuronsInLayers[i], configuration.NeuronsInLayers[i - 1],
                    configuration.Bias, activateFunc);
                prev.SetNextLayer(Layers[i - 1]);

                prev = Layers[i - 1];
            }

            _result = new ResultLayer();
            prev.SetNextLayer(_result);

            if (configuration.Code != null) SetCode(configuration.Code);

            _nnConfiguration = configuration;
        }

        private NN()
        {
        }

        public double[] Activate(double[] input)
        {
            Input.Activate(input);
            return _result.Result;
        }

        private double[] GetCode()
        {
            var result = new List<double>();
            foreach (var layer in Layers) result.AddRange(layer.GetCode());

            return result.ToArray();
        }

        private void SetCode(double[] code)
        {
            var listCode = code.ToList();
            foreach (var layer in Layers) layer.SetCode(listCode);
        }

        public NnConfiguration GetConfiguration()
        {
            _nnConfiguration.Code = GetCode();
            return _nnConfiguration;
        }
    }
}