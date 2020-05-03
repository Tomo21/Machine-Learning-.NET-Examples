using MNIST_Numsharp_CPU.Logic;
using MNIST_Numsharp_CPU.Logic.Interfaces;
using System;
using System.IO;
using System.Threading.Tasks;

namespace MNIST_Numsharp_CPU
{
    class Program
    {
        static async Task Main(string[] args)
        {
            Console.WriteLine("Enter number of epoch:");
            int epochs;

            while(!int.TryParse(Console.ReadLine(), out epochs) || epochs < 1){
                Console.WriteLine("Please enter whole positive number!");
            }

            Console.WriteLine("Enter minibatch size:");
            int minibatchSize;

            while (!int.TryParse(Console.ReadLine(), out minibatchSize) || minibatchSize < 1)
            {
                Console.WriteLine("Please enter whole positive number!");
            }

            Console.WriteLine("Loading MNIST data...");

            ILoader loader = new Loader();
            var basePath = Directory.GetCurrentDirectory().Split("bin");
            var dataList = await loader.LoadDataAsync(basePath[0] + @"Data\mnist_handwritten_train.json.gz", basePath[0] + @"\Data\mnist_handwritten_test.json.gz");

            Console.WriteLine("MNIST data loaded...");
            Console.WriteLine("Starting training, this will take some time...");

            INetwork network = new Network();
            network.Init(dataList.trainData, dataList.testData, new int[] { 784, 30, 10 });
            network.Train(epochs, minibatchSize, 0.1);
            
            Console.WriteLine("Training complete...");

            Console.WriteLine("Do You want to train for another epoch(y/n)?");
            var decision = Console.ReadLine();

            while(decision.ToLower().Trim() == "y"){
                Console.WriteLine("Starting training, this will take some time...");
                network.Train(epochs, minibatchSize, 0.1);

                Console.WriteLine("Do You want to train for another epoch(y/n)?");
                decision = Console.ReadLine();
            }
        }

        
    }
}
