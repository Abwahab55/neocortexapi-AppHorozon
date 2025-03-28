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

            Console.WriteLine("Running Image Binarization and Encoding...");
            var binarizer = new ImageBinarizerSpatialPattern(trainingFolder);
            binarizer.Run();
            Console.WriteLine("Image Binarization and Encoding Completed.");

            var sdrFiles = Directory.GetFiles(sdrFolder, "*.txt");
            if (sdrFiles.Length == 0)
            {
                Console.WriteLine($"Error: No SDR files found in '{sdrFolder}'. Exiting...");
                return;
            }

            Console.WriteLine("Initializing Classifiers (HTM and k-NN)...");
            var htmClassifier = new HtmImageClassifier(64, 64);
            var knnClassifier = new KnnImageClassifier();

            TrainClassifier(htmClassifier, sdrFolder, isHtm: true);
            TrainClassifier(knnClassifier, sdrFolder, isHtm: false);

            Console.WriteLine("Running Image Reconstruction via Classifiers...");
            var htmReconstructor = new HtmImageReconstructor();
            htmReconstructor.RunReconstruction(sdrFolder, outputFolder, reconstructedSdrFolder, htmClassifier, 64, 64);

            var knnReconstructor = new KnnImageReconstructor(64, 64, k: 1);
            knnReconstructor.RunReconstruction(sdrFolder, outputFolder, reconstructedSdrFolder, knnClassifier);

            Console.WriteLine("Computing Similarity between Original and Reconstructed SDRs...");
            var similarityResults = CompareOriginalAndReconstructedSDRs(sdrFolder, reconstructedSdrFolder);

            Console.WriteLine("Processing Pipeline Completed.");

            GenerateSimilarityGraph(similarityResults);
        }

        private static void TrainClassifier(IClassifier<int[], string> classifier, string sdrFolder, bool isHtm)
        {
            Console.WriteLine($"Training Classifier: {classifier.GetType().Name}");

            var sdrFiles = Directory.GetFiles(sdrFolder, "*.txt")
                .Where(file => !file.EndsWith("_binarized.txt"))
                .OrderBy(x => x) // order = sequence
                .ToList();

            int trainingCycles = isHtm ? 20 : 1;

            for (int cycle = 0; cycle < trainingCycles; cycle++)
            {
                if (isHtm && classifier is HtmImageClassifier htm)
                    htm.ResetTemporalMemory(); // clear old context

                foreach (var sdrFile in sdrFiles)
                {
                    string fileName = Path.GetFileNameWithoutExtension(sdrFile);
                    int[] sdr = ReadSdrFromFile(sdrFile);
                    classifier.Learn(sdr, new Cell[sdr.Length]);

                    if (cycle == 0)
                        Console.WriteLine($"Trained on {fileName} (SDR length {sdr.Length})");
                }

                if (isHtm)
                    Console.WriteLine($"HTM Training Cycle {cycle + 1}/{trainingCycles} completed.");
            }

            Console.WriteLine("Training Completed.\n");
        }

        private static Dictionary<string, (double htmSim, double knnSim)> CompareOriginalAndReconstructedSDRs(string sdrFolder, string reconstructedSdrFolder)
        {
            var results = new Dictionary<string, (double htmSim, double knnSim)>();

            var originalFiles = Directory.GetFiles(sdrFolder, "*.txt").OrderBy(x => x).ToList();
            foreach (var origFile in originalFiles)
            {
                string name = Path.GetFileNameWithoutExtension(origFile);
                string htmReconFile = Path.Combine(reconstructedSdrFolder, $"{name}_HTM_Reconstructed.txt");
                string knnReconFile = Path.Combine(reconstructedSdrFolder, $"{name}_KNN_Reconstructed.txt");

                if (!File.Exists(htmReconFile) || !File.Exists(knnReconFile))
                    continue;

                int[] origSdr = ReadSdrFromFile(origFile);
                int[] htmReconSdr = ReadSdrFromFile(htmReconFile);
                int[] knnReconSdr = ReadSdrFromFile(knnReconFile);

                Console.WriteLine($"Similarity Results for \"{name}\":");

                Console.WriteLine(" [HTM Similarity Metrics]");
                PrintSimilarityMetrics(origSdr, htmReconSdr);

                Console.WriteLine(" [k-NN Similarity Metrics]");
                PrintSimilarityMetrics(origSdr, knnReconSdr);

                Console.WriteLine();

                double htmSim = ComputeHybridSimilarity(origSdr, htmReconSdr) * 100;
                double knnSim = ComputeHybridSimilarity(origSdr, knnReconSdr) * 100;

                results[name] = (htmSim, knnSim);
            }

            return results;
        }

        private static void PrintSimilarityMetrics(int[] original, int[] prediction)
        {
            double jaccard = MathHelpers.JaccardSimilarityofBinaryArrays(original, prediction) * 100;
            double cosine = ComputeCosineSimilarity(original, prediction) * 100;
            double hamming = ComputeHammingSimilarity(original, prediction) * 100;
            double hybrid = (jaccard + cosine + hamming) / 3.0;

            Console.WriteLine($"  Cosine:  {cosine :F4}");
            Console.WriteLine($"  Jaccard: {jaccard :F4}");
            Console.WriteLine($"  Hamming: {hamming :F4}");
            Console.WriteLine($"  Hybrid:  {hybrid :F4}");
        }

        private static void GenerateSimilarityGraph(Dictionary<string, (double htmSim, double knnSim)> similarityResults)
        {
            string outputFolder = Path.Combine(Environment.CurrentDirectory, "SimilarityPlots_Image_Inputs");
            EnsureDirectoryExists(outputFolder);

            int width = 1600;  // larger canvas
            int height = 600;
            var bmp = new Bitmap(width, height);
            using var g = Graphics.FromImage(bmp);
            g.Clear(Color.White);

            var font = new Font("Arial", 9);
            var labelFont = new Font("Arial", 8);
            var titleFont = new Font("Arial", 12, FontStyle.Bold);

            double maxSim = 100.0; // all similarity metrics max at 100%
            int barGroupCount = similarityResults.Count;
            int groupWidth = width / barGroupCount;
            int barWidth = groupWidth / 3;
            int baseLineY = height - 100;

            // Draw Y-axis
            g.DrawLine(Pens.Black, 50, baseLineY, width - 50, baseLineY);
            for (int i = 0; i <= 10; i++)
            {
                int y = baseLineY - (int)(i * 0.1 * (height - 150));
                g.DrawLine(Pens.LightGray, 50, y, width - 50, y);
                g.DrawString($"{i * 10}%", labelFont, Brushes.Black, 5, y - 6);
            }

            int x = 60;
            foreach (var result in similarityResults)
            {
                string name = result.Key;
                double htmSim = result.Value.htmSim;
                double knnSim = result.Value.knnSim;

                int htmHeight = (int)((htmSim / maxSim) * (height - 150));
                int knnHeight = (int)((knnSim / maxSim) * (height - 150));

                g.FillRectangle(Brushes.Blue, x, baseLineY - htmHeight, barWidth, htmHeight);
                g.FillRectangle(Brushes.Green, x + barWidth + 2, baseLineY - knnHeight, barWidth, knnHeight);

                // Labels above bars
                g.DrawString($"{htmSim:F1}%", labelFont, Brushes.Blue, x, baseLineY - htmHeight - 15);
                g.DrawString($"{knnSim:F1}%", labelFont, Brushes.Green, x + barWidth + 2, baseLineY - knnHeight - 15);

                // X-label rotated for better spacing
                g.TranslateTransform(x, baseLineY + 10);
                g.RotateTransform(45);
                g.DrawString(name, font, Brushes.Black, 0, 0);
                g.ResetTransform();

                x += groupWidth;
            }

            // Title
            g.DrawString("HTM vs k-NN Image Reconstruction Similarity (Hybrid %)", titleFont, Brushes.Black, width / 3, 10);

            string outputPath = Path.Combine(outputFolder, "SimilarityComparison_Improved.png");
            bmp.Save(outputPath);
            Console.WriteLine($"📊 Improved similarity graph saved to {outputPath}");
        }

        private static int[] ReadSdrFromFile(string path)
        {
            string content = File.ReadAllText(path).Trim();
            return content.Split(new[] { ',', '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries)
                          .Select(bit => int.Parse(bit.Trim()))
                          .ToArray();
        }

        private static double ComputeHybridSimilarity(int[] sdr1, int[] sdr2)
        {
            if (sdr1.Length != sdr2.Length)
                throw new ArgumentException("SDRs must be the same length");

            double jaccardSim = MathHelpers.JaccardSimilarityofBinaryArrays(sdr1, sdr2);
            double cosSim = ComputeCosineSimilarity(sdr1, sdr2);
            double hammingSim = ComputeHammingSimilarity(sdr1, sdr2);

            return (jaccardSim + cosSim + hammingSim) / 3.0;
        }

        private static double ComputeCosineSimilarity(int[] sdr1, int[] sdr2)
        {
            double dot = sdr1.Zip(sdr2, (a, b) => a * b).Sum();
            double magA = Math.Sqrt(sdr1.Sum(a => a * a));
            double magB = Math.Sqrt(sdr2.Sum(b => b * b));
            return (magA == 0 || magB == 0) ? 0.0 : dot / (magA * magB);
        }

        private static double ComputeHammingSimilarity(int[] sdr1, int[] sdr2)
        {
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
