using System;
using System.Collections.Generic;

namespace N3K0NN.NeuralNetwork
{
    public static class ActivateFuncs
    {
        public static readonly Dictionary<Func<double, double>, Func<double, double>> Derivatives =
            new Dictionary<Func<double, double>, Func<double, double>>()
            {
                {LeakyReLu, LeakyReLuDerivative},
                {LogisticFunc, LogisticFuncDerivative}
            };

        public static double LeakyReLu(double x)
        {
            return x < 0 ? 0.01 * x : x;
        }

        public static double LeakyReLuDerivative(double x)
        {
            return x < 0 ? 0.01 : 1;
        }


        public static double LogisticFunc(double x)
        {
            return 1 / (1 + Math.Pow(Math.E, -x));
        }

        public static double LogisticFuncDerivative(double x)
        {
            return x * (1 - x);
        }
    }
}