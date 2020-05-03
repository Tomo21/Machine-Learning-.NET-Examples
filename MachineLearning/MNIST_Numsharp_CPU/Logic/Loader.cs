using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Text.Json;
using System.Threading.Tasks;
using MNIST_Numsharp_CPU.Logic.DeserializedObjects;
using MNIST_Numsharp_CPU.Logic.Interfaces;

namespace MNIST_Numsharp_CPU.Logic
{
    public class Loader : ILoader
    {
        private (List<Data> trainData, List<Data> testData) _data;

        public async Task<(List<Data> trainData, List<Data> testData)> LoadDataAsync(string trainFilePath, string testFilePath)
        {
            if (_data.testData != null && _data.trainData != null)
                return _data;

            using var str = new FileStream(trainFilePath, FileMode.Open, FileAccess.Read);
            using var uncompressed = new GZipStream(str, CompressionMode.Decompress);
            using var mem = new MemoryStream();
            await uncompressed.CopyToAsync(mem);
            mem.Position = 0;

            var trainingList = await JsonSerializer.DeserializeAsync<List<Data>>(mem);

            using var strTest = new FileStream(testFilePath, FileMode.Open, FileAccess.Read);
            using var uncompressedTest = new GZipStream(strTest, CompressionMode.Decompress);
            using var memTest = new MemoryStream();
            await uncompressedTest.CopyToAsync(memTest);
            memTest.Position = 0;

            var testList = await JsonSerializer.DeserializeAsync<List<Data>>(memTest);

            _data = (trainingList, testList);

            return _data;
        }

    }
}
