using System;
using System.Drawing;
using System.IO;
using System.Linq;
using NeoCortexApi.Classifiers;
using System.Collections.Generic;

namespace NeoCortexApiSample
{
    // This class handles the image reconstruction process using a k-NN classifier.
    // It predicts SDRs based on input SDRs and generates corresponding images.
    public class KnnImageReconstructor
    {
        private readonly int imageWidth;
        private readonly int imageHeight;
        private readonly int k; // Number of nearest neighbors to consider

        // Constructor to set image dimensions and value of k
        public KnnImageReconstructor(int width = 64, int height = 64, int k = 5)
        {
            imageWidth = width;
            imageHeight = height;
            this.k = k;
        }

        // Main method to perform reconstruction using input SDRs and a k-NN classifier
        public void RunReconstruction(string sdrFolder, string outputImageFolder,
                                      string reconstructedSdrFolder,
                                      IClassifier<int[], string> classifier)
        {
            // Ensure output folders exist
            Directory.CreateDirectory(outputImageFolder);
            Directory.CreateDirectory(reconstructedSdrFolder);

            // Filter out original SDR files (excluding binarized ones)
            var sdrFiles = Directory.GetFiles(sdrFolder, "*.txt")
                .Where(file => !file.EndsWith("_binarized.txt")).ToArray();

            foreach (var sdrFile in sdrFiles)
            {
                string name = Path.GetFileNameWithoutExtension(sdrFile);
                string outImagePath = Path.Combine(outputImageFolder, $"{name}_KNN_Reconstructed.png");
                string outSdrPath = Path.Combine(reconstructedSdrFolder, $"{name}_KNN_Reconstructed.txt");

                try
                {
                    // Read original SDR from text file
                    int[] originalSdr = File.ReadAllText(sdrFile)
                        .Split(new[] { ',', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries)
                        .Select(int.Parse).ToArray();

                    // Use the classifier to predict k closest SDRs
                    var predictions = classifier.GetPredictedInputValues(originalSdr, (short)k);

                    // Perform weighted voting over k predictions to get a reconstructed SDR
                    int[] reconstructedSdr = predictions.Count > 0
                        ? WeightedVoting(predictions)
                        : originalSdr; // Fallback to original if no prediction is returned

                    // Optionally, add noise to reduce overfitting and simulate imperfections
                    reconstructedSdr = AddNoiseToSdr(reconstructedSdr, noiseBits: 400);

                    // Save the reconstructed SDR as a text file
                    File.WriteAllText(outSdrPath, string.Join(",", reconstructedSdr));

                    // Create and save the visual image based on the reconstructed SDR
                    using (Bitmap bmp = new Bitmap(imageWidth, imageHeight))
                    {
                        for (int i = 0; i < reconstructedSdr.Length; i++)
                        {
                            int x = i % imageWidth;
                            int y = i / imageWidth;

                            // Use gray color to indicate differences from original SDR
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

        // Combines predictions from multiple neighbors using similarity-weighted voting
        private int[] WeightedVoting(List<ClassifierResult<int[]>> predictions)
        {
            int length = predictions[0].PredictedInput.Length;
            int[] votedSdr = new int[length];

            for (int i = 0; i < length; i++)
            {
                // Score for bits set to 1 based on similarity weights
                double onesScore = predictions
                    .Where(p => p.PredictedInput[i] == 1)
                    .Sum(p => p.Similarity);

                // Score for bits set to 0
                double zerosScore = predictions
                    .Where(p => p.PredictedInput[i] == 0)
                    .Sum(p => p.Similarity);

                // Choose the bit with the higher cumulative similarity
                votedSdr[i] = onesScore >= zerosScore ? 1 : 0;
            }

            return votedSdr;
        }

        // Randomly flip a fixed number of bits in the SDR to simulate noise or prediction error
        private static int[] AddNoiseToSdr(int[] sdr, int noiseBits = 20)
        {
            var rand = new Random();
            var noisy = (int[])sdr.Clone();

            // Randomly pick bit positions to flip
            var indices = Enumerable.Range(0, sdr.Length)
                                    .OrderBy(_ => rand.Next())
                                    .Take(noiseBits);

            foreach (int i in indices)
                noisy[i] = noisy[i] == 1 ? 0 : 1;

            return noisy;
        }
    }
}
