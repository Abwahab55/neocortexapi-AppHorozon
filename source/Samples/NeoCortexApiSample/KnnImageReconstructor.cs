using System;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using NeoCortexApi.Classifiers;
using NeoCortexApi.Entities;

namespace NeoCortexApiSample
{
    /// <summary>
    /// Handles image reconstruction using the k-NN classifier. It finds the k nearest stored SDRs for each input and combines them to reconstruct the image.
    /// Saves reconstructed SDRs and corresponding images (with differences highlighted).
    /// </summary>
    public class KnnImageReconstructor
    {
        private int imageWidth;
        private int imageHeight;
        private int _k;

        public KnnImageReconstructor(int width = 64, int height = 64, int k = 5)
        {
            imageWidth = width;
            imageHeight = height;
            _k = k; // number of nearest neighbors to use for reconstruction voting
        }

        /// <summary>
        /// Runs the reconstruction for all SDR files in sdrFolder using the k-NN classifier.
        /// Saves output images to outputImageFolder and output SDR files to reconstructedSdrFolder.
        /// </summary>
        public void RunReconstruction(string sdrFolder, string outputImageFolder,
                                      string reconstructedSdrFolder,
                                      IClassifier<int[], string> classifier)
        {
            if (!Directory.Exists(outputImageFolder))
                Directory.CreateDirectory(outputImageFolder);
            if (!Directory.Exists(reconstructedSdrFolder))
                Directory.CreateDirectory(reconstructedSdrFolder);

            string[] sdrFiles = Directory.GetFiles(sdrFolder, "*.txt");
            if (sdrFiles.Length == 0)
            {
                Console.WriteLine("No SDR files found. Ensure SDR generation was successful.");
                return;
            }

            foreach (var sdrFile in sdrFiles)
            {
                string name = Path.GetFileNameWithoutExtension(sdrFile);
                string outImagePath = Path.Combine(outputImageFolder, $"{name}_KNN_Reconstructed.png");
                string outSdrPath = Path.Combine(reconstructedSdrFolder, $"{name}_KNN_Reconstructed.txt");

                try
                {
                    int[] originalSdr = File.ReadAllText(sdrFile).Trim()
                                             .Replace("\r", "").Replace("\n", "")
                                             .Split(',')
                                             .Where(x => !string.IsNullOrWhiteSpace(x))
                                             .Select(int.Parse)
                                             .ToArray();

                    // Get the top k most similar SDRs from the classifier
                    var predictions = classifier.GetPredictedInputValues(originalSdr, (short)_k);
                    // Combine the top k predictions using weighted voting to form the final reconstructed SDR
                    int[] reconstructedSdr = (predictions.Count > 0) ? WeightedVoting(predictions) : originalSdr;

                    File.WriteAllText(outSdrPath, string.Join(",", reconstructedSdr));
                    SaveSdrToImage(reconstructedSdr, originalSdr, outImagePath);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error reconstructing {name}: {ex.Message}");
                }
            }
        }

        /// <summary>
        /// Performs a weighted vote across multiple predicted SDRs to decide the final bit values.
        /// Each prediction's similarity score serves as the weight for its bits.
        /// </summary>
        private int[] WeightedVoting(List<ClassifierResult<int[]>> predictions)
        {
            int length = predictions[0].PredictedInput.Length;
            int[] finalSdr = new int[length];
            // Sum the similarity scores for '1' bits and '0' bits at each position across all predictions
            for (int i = 0; i < length; i++)
            {
                double onesSum = predictions.Where(p => p.PredictedInput[i] == 1).Sum(p => p.Similarity);
                double zerosSum = predictions.Where(p => p.PredictedInput[i] == 0).Sum(p => p.Similarity);
                // If the weighted sum for ones is greater, choose 1; otherwise choose 0.
                finalSdr[i] = onesSum > zerosSum ? 1 : 0;
            }
            return finalSdr;
        }

        /// <summary>
        /// Saves a reconstructed SDR as an image file (PNG). 
        /// Black pixels indicate correctly reconstructed 1-bits, white pixels indicate correctly reconstructed 0-bits, and gray pixels indicate differences.
        /// </summary>
        private void SaveSdrToImage(int[] reconstructedSdr, int[] originalSdr, string outputPath)
        {
            using Bitmap bmp = new Bitmap(imageWidth, imageHeight);
            for (int i = 0; i < Math.Min(reconstructedSdr.Length, originalSdr.Length); i++)
            {
                int x = i % imageWidth;
                int y = i / imageWidth;
                int bitRecon = reconstructedSdr[i];
                int bitOrig = originalSdr[i];
                Color color = (bitRecon == bitOrig)
                              ? (bitRecon == 1 ? Color.Black : Color.White)
                              : Color.Gray;
                bmp.SetPixel(x, y, color);
            }
            bmp.Save(outputPath);
        }
    }
}
