using System;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Vision;

namespace ML_NET_Example_GPU
{
    class Program
    {
        static void Main(string[] args)
        {
            var imagesFolder = Path.Combine(Environment.CurrentDirectory, "..", "..", "..", "images");
            var files = Directory.GetFiles(imagesFolder, "*", SearchOption.AllDirectories);

            var images = files.Select(file => new ImageData
            {
                Image = File.ReadAllBytes(file),
                Label = Directory.GetParent(file).Name
            });

            var context = new MLContext();

            var imageData = context.Data.LoadFromEnumerable(images);
            var imageDataShuffled = context.Data.ShuffleRows(imageData);

            imageDataShuffled = context.Transforms.Conversion
                    .MapValueToKey("Label", keyOrdinality: Microsoft.ML.Transforms
                    .ValueToKeyMappingEstimator.KeyOrdinality.ByValue)
                .Fit(imageDataShuffled)
                .Transform(imageDataShuffled);

            var testTrainData = context.Data.TrainTestSplit(imageDataShuffled, testFraction: 0.2);

            var options = new ImageClassificationTrainer.Options()
            {
                FeatureColumnName = "Image",
                LabelColumnName = "Label",
                Arch = ImageClassificationTrainer.Architecture.ResnetV2101,
                Epoch = 100,
                BatchSize = 10,
                MetricsCallback = (metrics) => Console.WriteLine(metrics),
                ValidationSetFraction = 0.2f,
            };

            var pipeline = context.MulticlassClassification.Trainers.
                ImageClassification(options)
                .Append(context.Transforms.Conversion.MapKeyToValue(
                    outputColumnName: "PredictedLabel",
                    inputColumnName: "PredictedLabel"));

            var model = pipeline.Fit(testTrainData.TrainSet);

            //Console.WriteLine("Training with transfer learning finished.");

            var predicions = model.Transform(testTrainData.TestSet);

            var metrics = context.MulticlassClassification.Evaluate(predicions);

            Console.WriteLine(Environment.NewLine);
            Console.WriteLine($"Log loss - {metrics.LogLoss}");

            var predictionEngine = context.Model.CreatePredictionEngine<ImageData, ImagePrediction>(model);

            var testImagesFolder = Path.Combine(Environment.CurrentDirectory, "..", "..", "..", "test");

            var testFiles = Directory.GetFiles(testImagesFolder, "*", SearchOption.AllDirectories);

            var testImages = testFiles.Select(file => new ImageData
            {
                Image = File.ReadAllBytes(file),
                Label = Directory.GetParent(file).Name
            });

            Console.WriteLine(Environment.NewLine);

            foreach (var image in testImages)
            {
                var prediction = predictionEngine.Predict(image);

                Console.WriteLine($"Image : {image.Label}, Score : {prediction.Score.Max()}, Predicted Label : {prediction.PredictedLabel}");
            }

            context.Model.Save(model, imageData.Schema, "./dnn_model.zip");

            Console.ReadLine();
        }
    }
}
