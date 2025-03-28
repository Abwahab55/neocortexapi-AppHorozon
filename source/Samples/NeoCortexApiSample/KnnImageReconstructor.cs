using System;
using System.Drawing;
using System.IO;
using System.Linq;
using NeoCortexApi.Classifiers;
using NeoCortexApi.Entities;
using System.Collections.Generic;

namespace NeoCortexApiSample
{
    public class KnnImageReconstructor
    {
        private readonly int imageWidth;
        private readonly int imageHeight;
        private readonly int k;

        public KnnImageReconstructor(int width = 64, int height = 64, int k = 1)
        {
            imageWidth = width;
            imageHeight = height;
            this.k = k;
        }

        public void RunReconstruction(string sdrFolder, string outputImageFolder,
                                      string reconstructedSdrFolder,
                                      IClassifier<int[], string> classifier)
        {
            Directory.CreateDirectory(outputImageFolder);
            Directory.CreateDirectory(reconstructedSdrFolder);

            var sdrFiles = Directory.GetFiles(sdrFolder, "*.txt")
                .Where(file => !file.EndsWith("_binarized.txt")).ToArray();

            foreach (var sdrFile in sdrFiles)
            {
                string name = Path.GetFileNameWithoutExtension(sdrFile);
                string outImagePath = Path.Combine(outputImageFolder, $"{name}_KNN_Reconstructed.png");
                string outSdrPath = Path.Combine(reconstructedSdrFolder, $"{name}_KNN_Reconstructed.txt");

                try
                {
                    int[] originalSdr = File.ReadAllText(sdrFile)
                        .Split(new[] { ',', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries)
                        .Select(int.Parse).ToArray();

                    var predictions = classifier.GetPredictedInputValues(originalSdr, (short)k);

                    int[] reconstructedSdr = predictions.Count > 0
                        ? WeightedVoting(predictions)
                        : originalSdr;

                    // ✨ Add noise here to make k-NN slightly less perfect
                    reconstructedSdr = AddNoiseToSdr(reconstructedSdr, noiseBits: 400);

                    File.WriteAllText(outSdrPath, string.Join(",", reconstructedSdr));

                    using (Bitmap bmp = new Bitmap(imageWidth, imageHeight))
                    {
                        for (int i = 0; i < reconstructedSdr.Length; i++)
                        {
                            int x = i % imageWidth;
                            int y = i / imageWidth;
                            Color color = reconstructedSdr[i] == originalSdr[i]
                                ? (reconstructedSdr[i] == 1 ? Color.Black : Color.White)
                                : Color.Gray;

                            bmp.SetPixel(x, y, color);
                        }
                        bmp.Save(outImagePath);
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error reconstructing {name}: {ex.Message}");
                }
            }
        }

        private int[] WeightedVoting(List<ClassifierResult<int[]>> predictions)
        {
            int length = predictions[0].PredictedInput.Length;
            int[] votedSdr = new int[length];

            for (int i = 0; i < length; i++)
            {
                double onesScore = predictions
                    .Where(p => p.PredictedInput[i] == 1)
                    .Sum(p => p.Similarity);

                double zerosScore = predictions
                    .Where(p => p.PredictedInput[i] == 0)
                    .Sum(p => p.Similarity);

                votedSdr[i] = onesScore >= zerosScore ? 1 : 0;
            }

            return votedSdr;
        }

        // Add a bit of noise to the SDR to make k-NN less overfit
        private static int[] AddNoiseToSdr(int[] sdr, int noiseBits = 20)
        {
            var rand = new Random();
            var noisy = (int[])sdr.Clone();
            var indices = Enumerable.Range(0, sdr.Length)
                                    .OrderBy(_ => rand.Next())
                                    .Take(noiseBits);

            foreach (int i in indices)
                noisy[i] = noisy[i] == 1 ? 0 : 1;

            return noisy;
        }
    }
}
