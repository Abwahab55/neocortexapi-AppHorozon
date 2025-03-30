using System;
using System.Drawing;
using System.IO;
using System.Linq;
using NeoCortexApi.Classifiers;
using NeoCortexApi.Entities;
using System.Collections.Generic;

namespace NeoCortexApiSample
{
    /// <summary>
    /// Responsible for reconstructing images from SDRs using a k-NN classifier.
    /// The process includes loading SDRs, predicting reconstructed versions,
    /// saving new SDRs, and creating visual output as images.
    /// </summary>
    public class KnnImageReconstructor
    {
        private readonly int imageWidth;
        private readonly int imageHeight;
        private readonly int k;

        public KnnImageReconstructor(int width = 64, int height = 64, int k = 5)
        {
            imageWidth = width;
            imageHeight = height;
            this.k = k;
        }

        /// <summary>
        /// Reconstructs images by predicting SDRs using a k-NN classifier,
        /// adds noise for realism, and generates image files.
        /// </summary>
        public void RunReconstruction(string sdrFolder, string outputImageFolder,
                                      string reconstructedSdrFolder,
                                      IClassifier<int[], string> classifier)
        {
            Directory.CreateDirectory(outputImageFolder);
            Directory.CreateDirectory(reconstructedSdrFolder);

            // Load all SDR files, excluding binarized versions
            var sdrFiles = Directory.GetFiles(sdrFolder, "*.txt")
                .Where(file => !file.EndsWith("_binarized.txt")).ToArray();

            foreach (var sdrFile in sdrFiles)
            {
                string name = Path.GetFileNameWithoutExtension(sdrFile);
                string outImagePath = Path.Combine(outputImageFolder, $"{name}_KNN_Reconstructed.png");
                string outSdrPath = Path.Combine(reconstructedSdrFolder, $"{name}_KNN_Reconstructed.txt");

                try
                {
                    // Load original SDR from file
                    int[] originalSdr = File.ReadAllText(sdrFile)
                        .Split(new[] { ',', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries)
                        .Select(int.Parse).ToArray();

                    // Predict top-k closest SDRs using k-NN
                    var predictions = classifier.GetPredictedInputValues(originalSdr, (short)k);

                    // Reconstruct SDR via weighted voting
                    int[] reconstructedSdr = predictions.Count > 0
                        ? WeightedVoting(predictions)
                        : originalSdr;

                    // Optionally add noise to simulate imperfect predictions
                    reconstructedSdr = AddNoiseToSdr(reconstructedSdr, noiseBits: 400);

                    // Save reconstructed SDR to disk
                    File.WriteAllText(outSdrPath, string.Join(",", reconstructedSdr));

                    // Convert SDR to image and save as PNG
                    using (Bitmap bmp = new Bitmap(imageWidth, imageHeight))
                    {
                        for (int i = 0; i < reconstructedSdr.Length; i++)
                        {
                            int x = i % imageWidth;
                            int y = i / imageWidth;

                            // Visualize matching pixels as black/white, mismatches as gray
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

        /// <summary>
        /// Performs weighted voting across the top-k predictions.
        /// Each bit is set to 1 or 0 based on the total similarity score.
        /// </summary>
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

        /// <summary>
        /// Adds noise by flipping a fixed number of bits in the SDR.
        /// Useful to simulate slight variation and avoid perfect overlap.
        /// </summary>
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
