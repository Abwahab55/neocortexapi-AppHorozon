using System;
using System.IO;
using System.Linq;
using System.Drawing;
using NeoCortexApi.Classifiers;
using NeoCortexApi.Entities;
using NeoCortexApi.Utility;
using System.Collections.Generic;

namespace NeoCortexApiSample
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Starting Image Processing Pipeline...");
            string trainingFolder = Path.Combine(Environment.CurrentDirectory, @"..\..\..\Sample");

            string sdrFolder = Path.Combine(Environment.CurrentDirectory, "SDR_Values");
            string outputFolder = Path.Combine(Environment.CurrentDirectory, "ReconstructedImages");
            string reconstructedSdrFolder = Path.Combine(Environment.CurrentDirectory, "Reconstructed_SDRs");

            EnsureDirectoryExists(trainingFolder);
            EnsureDirectoryExists(sdrFolder);
            EnsureDirectoryExists(outputFolder);
            EnsureDirectoryExists(reconstructedSdrFolder);

            Console.WriteLine("Running Image Binarization...");

            var binarizer = new ImageBinarizerSpatialPattern(trainingFolder);
            binarizer.Run();

            Console.WriteLine("Image Binarization Completed.");

            var sdrFiles = Directory.GetFiles(sdrFolder, "*.txt");
            if (sdrFiles.Length == 0)
            {
                Console.WriteLine($"Error: No SDR files found in '{sdrFolder}'. Exiting...");
                return;
            }

            Console.WriteLine("Initializing Classifiers...");
            var htmClassifier = new HtmImageClassifier();
            var knnClassifier = new KnnImageClassifier();

            TrainClassifier(htmClassifier, sdrFolder, true);
            TrainClassifier(knnClassifier, sdrFolder, false);

            Console.WriteLine("Running Image Reconstruction via Classifiers...");
            var htmReconstructor = new HtmImageReconstructor();
            htmReconstructor.RunReconstruction(sdrFolder, outputFolder, reconstructedSdrFolder, htmClassifier);
            var knnReconstructor = new KnnImageReconstructor();
            knnReconstructor.RunReconstruction(sdrFolder, outputFolder, reconstructedSdrFolder, knnClassifier);

            Console.WriteLine("Computing Similarity between Original and Reconstructed SDRs...");
            var similarityResults = CompareOriginalAndReconstructedSDRs(sdrFolder, reconstructedSdrFolder);

            Console.WriteLine("Processing Pipeline Completed.");

            GenerateSimilarityGraph(similarityResults);
        }

        private static void TrainClassifier(IClassifier<int[], string> classifier, string sdrFolder, bool isHtm)
        {
            Console.WriteLine($"Training Classifier: {classifier.GetType().Name}");
            var sdrFiles = Directory.GetFiles(sdrFolder, "*.txt").OrderBy(x => x).ToList();
            int trainingCycles = isHtm ? 20 : 1;

            for (int cycle = 0; cycle < trainingCycles; cycle++)
            {
                foreach (var sdrFile in sdrFiles)
                {
                    string fileName = Path.GetFileNameWithoutExtension(sdrFile);
                    int[] sdr = ReadSdrFromFile(sdrFile);
                    classifier.Learn(sdr, new Cell[sdr.Length]);
                    if (cycle == 0)
                        Console.WriteLine($"Trained on {fileName} (SDR length {sdr.Length})");
                }
                if (isHtm)
                {
                    Console.WriteLine($"HTM Training Cycle {cycle + 1}/{trainingCycles} completed.");
                }
            }
            Console.WriteLine("Training Completed.");
        }

        private static void GenerateSimilarityGraph(Dictionary<string, (double htmSim, double knnSim)> similarityResults)
        {
            string outputFolder = Path.Combine(Environment.CurrentDirectory, "SimilarityPlots_Image_Inputs");

            EnsureDirectoryExists(outputFolder);

            int width = 800;
            int height = 400;
            var bmp = new Bitmap(width, height);
            var g = Graphics.FromImage(bmp);

            g.Clear(Color.White);
            var pen = new Pen(Color.Black);

            double maxHtmSim = similarityResults.Values.Max(r => r.htmSim);
            double maxKnnSim = similarityResults.Values.Max(r => r.knnSim);
            double maxSim = Math.Max(maxHtmSim, maxKnnSim);

            int barWidth = width / (2 * similarityResults.Count + 1);
            int padding = 10;
            int baseLineY = height - 50;

            int x = padding;
            foreach (var result in similarityResults)
            {
                string name = result.Key;
                double htmSim = result.Value.htmSim;
                double knnSim = result.Value.knnSim;

                int htmBarHeight = (int)((htmSim / maxSim) * (height - 50));
                g.FillRectangle(Brushes.Blue, x, baseLineY - htmBarHeight, barWidth, htmBarHeight);

                int knnBarHeight = (int)((knnSim / maxSim) * (height - 50));
                g.FillRectangle(Brushes.Green, x + barWidth, baseLineY - knnBarHeight, barWidth, knnBarHeight);

                g.DrawString(name, new Font("Arial", 8), Brushes.Black, new PointF(x, baseLineY + 5));

                g.DrawString("HTM", new Font("Arial", 10), Brushes.Blue, new PointF(x - 50, baseLineY - htmBarHeight - 15));
                g.DrawString("KNN", new Font("Arial", 10), Brushes.Green, new PointF(x + barWidth + 10, baseLineY - knnBarHeight - 15));

                x += 2 * barWidth + padding;
            }

            string outputPath = Path.Combine(outputFolder, "SimilarityComparison.png");
            bmp.Save(outputPath);
            Console.WriteLine($"Similarity graph saved to {outputPath}.");
        }

        private static Dictionary<string, (double htmSim, double knnSim)> CompareOriginalAndReconstructedSDRs(string sdrFolder, string reconstructedSdrFolder)
        {
            var originalFiles = Directory.GetFiles(sdrFolder, "*.txt").OrderBy(x => x).ToList();
            var similarityResults = new Dictionary<string, (double htmSim, double knnSim)>();

            foreach (var origFile in originalFiles)
            {
                string name = Path.GetFileNameWithoutExtension(origFile);
                string htmReconFile = Path.Combine(reconstructedSdrFolder, $"{name}_HTM_Reconstructed.txt");
                string knnReconFile = Path.Combine(reconstructedSdrFolder, $"{name}_KNN_Reconstructed.txt");

                if (!File.Exists(htmReconFile) || !File.Exists(knnReconFile))
                {
                    Console.WriteLine($"Missing reconstructed SDRs for {name}. Skipping comparison.");
                    continue;
                }

                int[] origSdr = ReadSdrFromFile(origFile);
                int[] htmReconSdr = ReadSdrFromFile(htmReconFile);
                int[] knnReconSdr = ReadSdrFromFile(knnReconFile);

                double htmSim = ComputeHybridSimilarity(origSdr, htmReconSdr) * 100;
                double knnSim = ComputeHybridSimilarity(origSdr, knnReconSdr) * 100;

                similarityResults[name] = (htmSim, knnSim);
                Console.WriteLine($"Similarity Results for {name}:");
                Console.WriteLine($" HTM Similarity: {htmSim:0.00}%");
                Console.WriteLine($" KNN Similarity: {knnSim:0.00}%");
            }

            return similarityResults;
        }

        private static int[] ReadSdrFromFile(string path)
        {
            return File.ReadAllText(path).Trim()
                       .Split(',')
                       .Where(str => str != "")
                       .Select(str => int.TryParse(str, out int bit) ? bit : 0)
                       .ToArray();
        }

        private static double ComputeHybridSimilarity(int[] sdr1, int[] sdr2)
        {
            if (sdr1.Length != sdr2.Length)
                throw new ArgumentException("SDRs must be the same length");

            double jaccardSim = MathHelpers.JaccardSimilarityofBinaryArrays(sdr1, sdr2);
            double dot = sdr1.Zip(sdr2, (a, b) => a * b).Sum();
            double magA = Math.Sqrt(sdr1.Sum(a => a * a));
            double magB = Math.Sqrt(sdr2.Sum(b => b * b));
            double cosSim = (magA == 0 || magB == 0) ? 0.0 : dot / (magA * magB);
            double hammingSim = ComputeHammingSimilarity(sdr1, sdr2);

            return (jaccardSim + cosSim + hammingSim) / 3.0;
        }

        private static double ComputeHammingSimilarity(int[] sdr1, int[] sdr2)
        {
            if (sdr1.Length != sdr2.Length)
                throw new ArgumentException("SDRs must be the same length");

            int matchingBits = sdr1.Zip(sdr2, (a, b) => a == b ? 1 : 0).Sum();
            return (double)matchingBits / sdr1.Length;
        }

        private static void EnsureDirectoryExists(string path)
        {
            if (!Directory.Exists(path))
                Directory.CreateDirectory(path);
        }
    }
}