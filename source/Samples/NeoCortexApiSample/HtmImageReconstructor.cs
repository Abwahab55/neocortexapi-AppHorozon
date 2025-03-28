using System;
using System.Drawing;
using System.IO;
using System.Linq;
using NeoCortexApi.Classifiers;
using NeoCortexApi.Entities;

namespace NeoCortexApiSample
{
    /// <summary>
    /// Responsible for reconstructing images using an HTM-based classifier.
    /// It loads SDRs, predicts reconstructed SDRs using Temporal Memory, and writes output images.
    /// </summary>
    public class HtmImageReconstructor
    {
        /// <summary>
        /// Reconstructs images from SDRs using HTM's temporal sequence learning.
        /// </summary>
        /// <param name="sdrFolder">Path to folder containing encoded SDR files.</param>
        /// <param name="outputImageFolder">Folder to save reconstructed images as PNGs.</param>
        /// <param name="reconstructedSdrFolder">Folder to save reconstructed SDRs as .txt.</param>
        /// <param name="classifier">HTM-based classifier used for prediction.</param>
        /// <param name="imageWidth">Width of the image (default 64).</param>
        /// <param name="imageHeight">Height of the image (default 64).</param>
        public void RunReconstruction(string sdrFolder, string outputImageFolder,
                                      string reconstructedSdrFolder,
                                      IClassifier<int[], string> classifier,
                                      int imageWidth = 64, int imageHeight = 64)
        {
            // Ensure output folders exist
            Directory.CreateDirectory(outputImageFolder);
            Directory.CreateDirectory(reconstructedSdrFolder);

            // Get all SDR files excluding binarized versions
            var sdrFiles = Directory.GetFiles(sdrFolder, "*.txt")
                                    .Where(file => !file.EndsWith("_binarized.txt"))
                                    .OrderBy(x => x) // maintain order for sequence learning
                                    .ToArray();

            // Cast to HtmImageClassifier to access sequence-related features
            var htm = classifier as HtmImageClassifier;
            htm?.ResetTemporalMemory(); // clear old memory context before reconstructing

            foreach (var sdrFile in sdrFiles)
            {
                string name = Path.GetFileNameWithoutExtension(sdrFile);
                string outImagePath = Path.Combine(outputImageFolder, $"{name}_HTM_Reconstructed.png");
                string outSdrPath = Path.Combine(reconstructedSdrFolder, $"{name}_HTM_Reconstructed.txt");

                try
                {
                    // Load the original SDR from file
                    int[] originalSdr = File.ReadAllText(sdrFile)
                        .Split(new[] { ',', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries)
                        .Select(int.Parse).ToArray();

                    // Feed SDR into HTM to learn it as part of a sequence
                    htm?.Learn(originalSdr, new Cell[originalSdr.Length]);

                    // Predict the next input SDR based on current temporal state
                    var predictions = htm?.GetPredictedInputValues(originalSdr, 1);
                    int[] reconstructedSdr = predictions?.FirstOrDefault()?.PredictedInput ?? originalSdr;

                    // Save reconstructed SDR to disk
                    File.WriteAllText(outSdrPath, string.Join(",", reconstructedSdr));

                    // Create and save image based on the predicted SDR
                    using (Bitmap bmp = new Bitmap(imageWidth, imageHeight))
                    {
                        for (int i = 0; i < reconstructedSdr.Length; i++)
                        {
                            int x = i % imageWidth;
                            int y = i / imageWidth;

                            // Match = black/white, Mismatch = gray
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
    }
}
