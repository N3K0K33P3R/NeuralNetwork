namespace N3K0NN.NeuralNetwork
{
    public interface IDataSet<in T>
    {
        double[] GetData(T data);
    }
}