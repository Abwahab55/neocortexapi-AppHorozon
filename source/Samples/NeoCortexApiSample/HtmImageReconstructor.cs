// HtmImageReconstructor.cs
using System;
using System.Drawing;
using System.IO;
using System.Linq;
using NeoCortexApi.Classifiers;
using NeoCortexApiSample;  // to access IClassifier implementations

namespace NeoCortexApiSample
{
    public class HtmImageReconstructor
    {
        public void RunReconstruction(string sdrFolder, string outputImageFolder,
                                      string reconstructedSdrFolder,
                                      IClassifier<int[], string> classifier,
                                      int imageWidth = 28, int imageHeight = 28)
        {
            if (!Directory.Exists(outputImageFolder))
                Directory.CreateDirectory(outputImageFolder);
            if (!Directory.Exists(reconstructedSdrFolder))
                Directory.CreateDirectory(reconstructedSdrFolder);

            var sdrFiles = Directory.GetFiles(sdrFolder, "*.txt");
            foreach (var sdrFile in sdrFiles)
            {
                string name = Path.GetFileNameWithoutExtension(sdrFile);
                string outImagePath = Path.Combine(outputImageFolder, $"{name}_HTM_Reconstructed.png");
                string outSdrPath = Path.Combine(reconstructedSdrFolder, $"{name}_HTM_Reconstructed.txt");

                // Read original SDR
                int[] originalSdr = File.ReadAllText(sdrFile).Trim()
                                        .Split(',').Where(x => x != "")
                                        .Select(int.Parse).ToArray();
                // Use classifier to predict the closest stored SDR (excluding identical)
                var predictions = classifier.GetPredictedInputValues(originalSdr, howMany: 1);
                int[] reconstructedSdr;
                if (predictions.Count > 0)
                    reconstructedSdr = predictions[0].PredictedInput;
                else
                    reconstructedSdr = originalSdr; // fallback (e.g., if only one pattern in training)

                // Save reconstructed SDR values to file
                File.WriteAllText(outSdrPath, string.Join(",", reconstructedSdr));

                // Create an image visualizing the reconstructed SDR vs original
                using (Bitmap bmp = new Bitmap(imageWidth, imageHeight))
                {
                    for (int i = 0; i < reconstructedSdr.Length; i++)
                    {
                        int x = i % imageWidth;
                        int y = i / imageWidth;
                        int bitRecon = reconstructedSdr[i];
                        int bitOrig = originalSdr.Length > i ? originalSdr[i] : 0;
                        Color color;
                        if (bitRecon == bitOrig)
                        {
                            // Match: black for 1, white for 0
                            color = (bitRecon == 1) ? Color.Black : Color.White;
                        }
                        else
                        {
                            // Mismatch: gray pixel
                            color = Color.FromArgb(128, 128, 128);
                        }
                        bmp.SetPixel(x, y, color);
                    }
                    bmp.Save(outImagePath);
                }
                Console.WriteLine($"Reconstructed (HTM) SDR saved: {outSdrPath}");
                Console.WriteLine($"Reconstructed (HTM) Image saved: {outImagePath}");
            }
        }
    }
}
