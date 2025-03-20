using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using NeoCortexApi.Classifiers;

namespace NeoCortexApiSample
{
    public class KnnImageReconstructor
    {
        private int imageWidth;
        private int imageHeight;
        private int _k;

        public KnnImageReconstructor(int width = 28, int height = 28, int k = 5)
        {
            imageWidth = width;
            imageHeight = height;
            _k = k;
        }

        public void RunReconstruction(string sdrFolder, string outputImageFolder,
                                      string reconstructedSdrFolder,
                                      IClassifier<int[], string> classifier)
        {
            if (!Directory.Exists(outputImageFolder))
                Directory.CreateDirectory(outputImageFolder);
            if (!Directory.Exists(reconstructedSdrFolder))
                Directory.CreateDirectory(reconstructedSdrFolder);

            var sdrFiles = Directory.GetFiles(sdrFolder, "*.txt");
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
                                            .Replace("\n", "").Replace("\r", "")
                                            .Split(',')
                                            .Where(x => !string.IsNullOrWhiteSpace(x))
                                            .Select(int.Parse)
                                            .ToArray();

                    var predictions = classifier.GetPredictedInputValues(originalSdr, (short)_k);
                    int[] reconstructedSdr = predictions.Count > 0 ? WeightedVoting(predictions) : originalSdr;

                    File.WriteAllText(outSdrPath, string.Join(",", reconstructedSdr));
                    SaveSdrToImage(reconstructedSdr, originalSdr, outImagePath);
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
            int[] finalSdr = new int[length];
            for (int i = 0; i < length; i++)
            {
                double onesSum = predictions.Where(p => p.PredictedInput[i] == 1).Sum(p => p.Similarity);
                double zerosSum = predictions.Where(p => p.PredictedInput[i] == 0).Sum(p => p.Similarity);
                finalSdr[i] = onesSum > zerosSum ? 1 : 0;
            }
            return finalSdr;
        }

        private void SaveSdrToImage(int[] reconstructedSdr, int[] originalSdr, string outputPath)
        {
            using (Bitmap bmp = new Bitmap(imageWidth, imageHeight))
            {
                for (int i = 0; i < Math.Min(reconstructedSdr.Length, originalSdr.Length); i++)
                {
                    int x = i % imageWidth;
                    int y = i / imageWidth;
                    int bitRecon = reconstructedSdr[i];
                    int bitOrig = originalSdr.Length > i ? originalSdr[i] : 0;
                    Color color = bitRecon == bitOrig ? (bitRecon == 1 ? Color.Black : Color.White) : Color.Gray;
                    bmp.SetPixel(x, y, color);
                }
                bmp.Save(outputPath);
            }
        }
    }
}
