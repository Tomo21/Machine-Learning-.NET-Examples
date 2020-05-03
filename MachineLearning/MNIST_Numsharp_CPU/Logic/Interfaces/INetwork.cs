using System.Collections.Generic;
using MNIST_Numsharp_CPU.Logic.DeserializedObjects;

namespace MNIST_Numsharp_CPU.Logic.Interfaces
{
    interface INetwork
    {
        public void Init(List<Data> trainData, List<Data> testData, int[] sizes);
        public void Train(int epochs, int mini_batch_size, double learningRate);
    }
}
