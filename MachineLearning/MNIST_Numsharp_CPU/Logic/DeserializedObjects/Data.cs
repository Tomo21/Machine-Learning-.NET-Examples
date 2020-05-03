using System.Text.Json.Serialization;
using NumSharp;

namespace MNIST_Numsharp_CPU.Logic.DeserializedObjects
{
    public class Data
    {
        private NDArray _vectorizedExpectedResult;
        private NDArray _reshapedImageBytes;

        [JsonPropertyName("label")]
        public int ExpectedResult { get; set; }

        [JsonPropertyName("image")]
        public short[] ImageBytes { get; set; }

        [JsonIgnore]
        public NDArray GetReshapedImageBytes
        {
            get
            {
                if (ImageBytes == null)
                    return null;

                if (_reshapedImageBytes == null)
                    _reshapedImageBytes = np.reshape(ImageBytes, new Shape(784, 1));

                return _reshapedImageBytes;
            }
        }

        [JsonIgnore]
        public NDArray VectorizedExpectedResult
        {
            get
            {
                if (_vectorizedExpectedResult == null)
                {
                    _vectorizedExpectedResult = np.zeros((10, 1));

                    _vectorizedExpectedResult[ExpectedResult] = 1.0;
                }

                return _vectorizedExpectedResult;
            }
        }
    }
}
