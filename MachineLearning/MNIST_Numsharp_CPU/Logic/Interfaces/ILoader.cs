using System.Collections.Generic;
using System.Threading.Tasks;
using MNIST_Numsharp_CPU.Logic.DeserializedObjects;

namespace MNIST_Numsharp_CPU.Logic.Interfaces
{
    public interface ILoader
    {
        public Task<(List<Data> trainData, List<Data> testData)> LoadDataAsync(string trainFilePath, string testFilePath);
    }
}
