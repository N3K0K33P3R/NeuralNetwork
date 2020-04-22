# NeuralNetwork
My implementation of simple neural network with back propagation learning method.
## How to:
### Use
Use `NN` class. Create a `NNConfiguration` and fill it (if you dont fill `Code` field, it will be initialized randomly). Then call `Activate` func, that returns array of outputs.
### Learn
Use `NNBackwards` class. Send it your `NN` and set the parameters. Also you should create `DataSet` class that implements `IDataSet` interface to get array of expected outputs from any object. Then you should call `Learn` function to activate back propagation. Don't forget to activate your `NN` with learning input before calling `Learn` function.
### Save\Load
Use `NeuralNetworkIo` class to save your NN and load `NNConfiguration`.
