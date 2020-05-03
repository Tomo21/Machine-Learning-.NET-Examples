using System;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Linq;
using System.Threading.Tasks;
using MNIST_Numsharp_CPU.Logic.DeserializedObjects;
using MNIST_Numsharp_CPU.Logic.Interfaces;
using NumSharp;

namespace MNIST_Numsharp_CPU.Logic
{
    public class Network : INetwork
    {
        #region Propertys
        private List<Data> TrainData { get; set; }
        private List<Data> TestData { get; set; }
        private int Num_layers { get; set; }
        private int[] Sizes { get; set; }
        private List<NDArray> Biases { get; set; } = new List<NDArray>();
        private List<NDArray> Weights { get; set; } = new List<NDArray>();
        #endregion

        public void Init(List<Data> trainData, List<Data> testData, int[] sizes)
        {
            TrainData = trainData;
            TestData = testData;

            Num_layers = sizes.Length;
            Sizes = sizes;

            for (int counter = 1; counter < Num_layers; counter++)
            {
                Biases.Add(np.random.randn(new int[] { Sizes[counter], 1 }));
            }

            foreach (var item in Sizes[..^1].Zip(Sizes[1..]))
            {
                Weights.Add(np.random.randn(new int[] { item.Second, item.First }));
            }
        }

        public void Train(int epochs, int miniBatchSize, double learningRate)
        {
            for (int counterEpochs = 0; counterEpochs < epochs; counterEpochs++)
            {
                var timeStart = DateTime.Now;

                var shuffled = TrainData.OrderBy(x => Guid.NewGuid()).ToList();

                for (int batchCounter = 0; batchCounter < TrainData.Count;
                    batchCounter += miniBatchSize)
                {
                    var miniBatch = shuffled.Skip(batchCounter).Take(miniBatchSize).ToList();
                    UpdateMiniBatch(miniBatch, learningRate);
                }

                var duration = DateTime.Now - timeStart;
                Console.WriteLine($"Epoch {counterEpochs + 1} completed...");
                Console.WriteLine($"Time to train this epoch: {duration}");

                Console.WriteLine("Starting network accuracy evaluation...");
                double numberOfCorrectPredictions = Evaluate();

                Console.WriteLine($"Accuracy percentage {numberOfCorrectPredictions}");
            }
        }

        private void UpdateMiniBatch(List<Data> miniBatch, double learningRate)
        {
            #region test for equality of results between multithreaded and single threaded 
            //var nabla_b2 = new List<NDArray>();
            //Biases.ForEach(x => { nabla_b2.Add(np.zeros(x.Shape, x.typecode)); });

            //var nabla_w2 = new List<NDArray>();
            //Weights.ForEach(x => { nabla_w2.Add(np.zeros(x.Shape, x.typecode)); });

            //miniBatch.ForEach(x =>
            //{
            //    var corrections = BackPropagation(x.GetReshapedImageBytes, x.VectorizedExpectedResult);

            //    for (int counter = 0; counter < nabla_b2.Count; counter++)
            //    {
            //        nabla_b2[counter] += corrections.biasCorrection[counter];
            //        nabla_w2[counter] += corrections.weightCorrection[counter];
            //    }
            //});
            #endregion

            var nabla_b = new List<NDArray>();
            Biases.ForEach(x => { nabla_b.Add(np.zeros(x.Shape, x.typecode)); });

            var nabla_w = new List<NDArray>();
            Weights.ForEach(x => { nabla_w.Add(np.zeros(x.Shape, x.typecode)); });

            //for some reason adding must be done in same order for results to be same as single threaded
            //(commutative law for addition not applicable because od bug in Numsharp)
            var correctionsBag = new ConcurrentDictionary<int, (List<NDArray> biasCorrection, List<NDArray> weightCorrection)>();

            Parallel.For(0, miniBatch.Count, x => {
                var corrections = BackPropagation(miniBatch[x].GetReshapedImageBytes, miniBatch[x].VectorizedExpectedResult);
                correctionsBag.TryAdd(x, corrections);
            });

            foreach (var item in correctionsBag.Values)
            {
                for (int counter = 0; counter < nabla_b.Count; counter++)
                {
                    nabla_b[counter] += item.biasCorrection[counter];
                    nabla_w[counter] += item.weightCorrection[counter];
                }
            }

            #region test for equality of results between multithreaded and single threaded - part 02
            //var a = np.Equals(nabla_b.ElementAt(0), nabla_b2.ElementAt(0));
            //var b = np.Equals(nabla_w.ElementAt(0), nabla_w2.ElementAt(0));
            #endregion

            for (int counter = 0; counter < nabla_b.Count; counter++)
            {
                Weights[counter] = Weights[counter] - (learningRate / miniBatch.Count) * nabla_w[counter];
                Biases[counter] = Biases[counter] - (learningRate / miniBatch.Count) * nabla_b[counter];
            }

        }

        private (List<NDArray> biasCorrection, List<NDArray> weightCorrection) BackPropagation(
                    NDArray input, NDArray expectedResult)
        {

            var nabla_b = new List<NDArray>();
            Biases.ForEach(x => { nabla_b.Add(np.zeros(x.Shape, x.typecode)); });

            var nabla_w = new List<NDArray>();
            Weights.ForEach(x => { nabla_w.Add(np.zeros(x.Shape, x.typecode)); });

            var activations = new List<NDArray>();
            activations.Add(input);

            var weightedInputs = new List<NDArray>();
            //feedfoward
            foreach (var item in Biases.Zip(Weights))
            {
                var z = np.dot(item.Second, activations[^1]) + item.First;
                weightedInputs.Add(z);
                activations.Add(Sigmoid(z));
            }

            //backpropagation
            var delta = CostDerivative(activations[^1], expectedResult)
                * SigmoidDerivation(weightedInputs[^1]);

            nabla_b[^1] = delta;
            nabla_w[^1] = np.dot(delta, activations[^2].transpose());

            for (int counter = 2; counter < Num_layers; counter++)
            {
                var sigDer = SigmoidDerivation(weightedInputs[^counter]);
                delta = np.dot(Weights[^(counter - 1)].transpose(), delta) * sigDer;

                nabla_b[^counter] = delta;
                nabla_w[^counter] = np.dot(delta, activations[^(counter + 1)].transpose());
            }

            return (nabla_b, nabla_w);
        }

        #region Utility

        private NDArray Sigmoid(NDArray z)
        {
            return 1.0 / (1.0 + np.exp(-z));
        }

        private NDArray SigmoidDerivation(NDArray z)
        {
            var sig = Sigmoid(z);
            return sig * (1 - sig);
        }

        private NDArray CostDerivative(NDArray outputActivations, NDArray expectedResult)
        {
            return outputActivations - expectedResult;
        }

        #region Accuracy Test
        private NDArray FeedFoward(NDArray input)
        {
            foreach (var item in Biases.Zip(Weights))
            {
                input = Sigmoid(np.dot(item.Second, input) + item.First);
            }

            return input;
        }

        private double Evaluate()
        {
            double numberOfCorrectPredictions = 0;
            var lockObj = new object();

            Parallel.ForEach(TestData, item => {
                var feedFowardResult = FeedFoward(item.GetReshapedImageBytes);
                var testResults = np.argmax(feedFowardResult);

                if (item.VectorizedExpectedResult[testResults] == 1)
                {
                    lock (lockObj)
                    {
                        numberOfCorrectPredictions += 1;
                    }
                }
            });

            return (numberOfCorrectPredictions / TestData.Count) * 100F;
        }
        #endregion

        #endregion

    }
}
