using System;
using System.Drawing;
using System.IO;
using System.Linq;
using NeoCortexApi.Classifiers;

namespace NeoCortexApiSample
{
    public class HtmImageReconstructor
    {
        //htm structure method
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

                try
                {
                    int[] originalSdr = File.ReadAllText(sdrFile).Trim()
                                        .Replace("\n", "").Replace("\r", "")
                                        .Split(',').Where(x => !string.IsNullOrWhiteSpace(x))
                                        .Select(int.Parse).ToArray();

                    var predictions = classifier.GetPredictedInputValues(originalSdr, howMany: 1);
                    int[] reconstructedSdr = predictions.Count > 0 ? predictions[0].PredictedInput : originalSdr;

                    File.WriteAllText(outSdrPath, string.Join(",", reconstructedSdr));

                    using (Bitmap bmp = new Bitmap(imageWidth, imageHeight))
                    {
                        for (int i = 0; i < reconstructedSdr.Length; i++)
                        {
                            int x = i % imageWidth;
                            int y = i / imageWidth;
                            int bitRecon = reconstructedSdr[i];
                            int bitOrig = originalSdr.Length > i ? originalSdr[i] : 0;
                            Color color = bitRecon == bitOrig ? (bitRecon == 1 ? Color.Black : Color.White) : Color.Gray;
                            bmp.SetPixel(x, y, color);
                        }
                        bmp.Save(outImagePath);
                    }
                }
                //exception handle
                catch (Exception ex)
                {
                    Console.WriteLine($"Error reconstructing {name}: {ex.Message}");
                }
            }
        }
    }
}
