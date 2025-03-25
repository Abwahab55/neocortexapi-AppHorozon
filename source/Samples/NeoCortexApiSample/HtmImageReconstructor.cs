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
                                      int imageWidth = 64, int imageHeight = 64)
        {
            Directory.CreateDirectory(outputImageFolder);
            Directory.CreateDirectory(reconstructedSdrFolder);

            var sdrFiles = Directory.GetFiles(sdrFolder, "*.txt")
                .Where(file => !file.EndsWith("_binarized.txt")).ToArray();

            foreach (var sdrFile in sdrFiles)
            {
                string name = Path.GetFileNameWithoutExtension(sdrFile);
                string outImagePath = Path.Combine(outputImageFolder, $"{name}_HTM_Reconstructed.png");
                string outSdrPath = Path.Combine(reconstructedSdrFolder, $"{name}_HTM_Reconstructed.txt");

                try
                {
                    int[] originalSdr = File.ReadAllText(sdrFile)
                        .Split(new[] { ',', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries)
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
                            Color color = reconstructedSdr[i] == originalSdr[i]
                                ? (reconstructedSdr[i] == 1 ? Color.Black : Color.White)
                                : Color.Gray;

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

