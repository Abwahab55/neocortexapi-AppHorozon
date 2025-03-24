using System;
using System.IO;
using System.Linq;
using System.Drawing;
using System.Collections.Generic;
using NeoCortexApi.Classifiers;
using NeoCortexApi.Entities;
using NeoCortexApi.Utility;

namespace NeoCortexApiSample
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Starting Image Processing Pipeline...");

            // Set up directories for input images and outputs (SDRs and reconstructed images)
            string trainingFolder = Path.Combine(Environment.CurrentDirectory, @"..\..\..\Sample");
            string sdrFolder = Path.Combine(Environment.CurrentDirectory, "SDR_Values");
            string outputFolder = Path.Combine(Environment.CurrentDirectory, "ReconstructedImages");
            string reconstructedSdrFolder = Path.Combine(Environment.CurrentDirectory, "Reconstructed_SDRs");

            // Ensure all necessary directories exist
            EnsureDirectoryExists(trainingFolder);
            EnsureDirectoryExists(sdrFolder);
            EnsureDirectoryExists(outputFolder);
            EnsureDirectoryExists(reconstructedSdrFolder);

            Console.WriteLine("Running Image Binarization and Encoding...");
            // Use Daenet's ImageBinarizer to convert images to binary and encode via Spatial Pooler
            var binarizer = new ImageBinarizerSpatialPattern(trainingFolder);
            binarizer.Run();
            Console.WriteLine("Image Binarization Completed. Encoded SDRs saved to " + sdrFolder);

            // Verify that SDR files were generated
            var sdrFiles = Directory.GetFiles(sdrFolder, "*.txt");
            if (sdrFiles.Length == 0)
            {
                Console.WriteLine($"Error: No SDR files found in '{sdrFolder}'. Exiting...");
                return;
            }

            Console.WriteLine("Initializing Classifiers (HTM and k-NN)...");
            // Instantiate classifiers that implement IClassifier<int[], string>
            IClassifier<int[], string> htmClassifier = new HtmImageClassifier();
            IClassifier<int[], string> knnClassifier = new KnnImageClassifier();

            // Train both classifiers on all SDRs. Use multiple cycles for HTM to allow sequence memory learning.
            TrainClassifier(htmClassifier, sdrFolder, isHtm: true);
            TrainClassifier(knnClassifier, sdrFolder, isHtm: false);

            Console.WriteLine("Running Image Reconstruction via Classifiers...");
            // Reconstruct images using each classifier and output results (SDR and image) for comparison
            var htmReconstructor = new HtmImageReconstructor();
            htmReconstructor.RunReconstruction(sdrFolder, outputFolder, reconstructedSdrFolder, htmClassifier, 64, 64);
            var knnReconstructor = new KnnImageReconstructor(64, 64, k: 5);
            knnReconstructor.RunReconstruction(sdrFolder, outputFolder, reconstructedSdrFolder, knnClassifier);

            Console.WriteLine("Computing Similarity between Original and Reconstructed SDRs...");
            // Compare original input SDRs with reconstructed SDRs from each classifier
            var similarityResults = CompareOriginalAndReconstructedSDRs(sdrFolder, reconstructedSdrFolder);

            Console.WriteLine("Processing Pipeline Completed.");
            // (Optional) Generate a bar chart image to visualize similarity comparisons
            GenerateSimilarityGraph(similarityResults);
        }

        /// <summary>
        /// Trains the given classifier on all SDR files in the specified folder.
        /// If isHtm is true, performs multiple training cycles to allow HTM Temporal Memory to learn.
        /// </summary>
        private static void TrainClassifier(IClassifier<int[], string> classifier, string sdrFolder, bool isHtm)
        {
            Console.WriteLine($"Training Classifier: {classifier.GetType().Name}");
            var sdrFiles = Directory.GetFiles(sdrFolder, "*.txt").OrderBy(x => x).ToList();
            int trainingCycles = isHtm ? 20 : 1;  // Train HTM in multiple passes for better sequence learning, k-NN just one pass

            for (int cycle = 0; cycle < trainingCycles; cycle++)
            {
                foreach (var sdrFile in sdrFiles)
                {
                    string fileName = Path.GetFileNameWithoutExtension(sdrFile);
                    int[] sdr = ReadSdrFromFile(sdrFile);
                    // Learn this SDR pattern (for HTM, output cells are not used in this simple reconstruction scenario)
                    classifier.Learn(sdr, new Cell[sdr.Length]);
                    if (cycle == 0)
                        Console.WriteLine($"Trained on {fileName} (SDR length {sdr.Length})");
                }
                if (isHtm)
                {
                    Console.WriteLine($"HTM Training Cycle {cycle + 1}/{trainingCycles} completed.");
                }
            }
            Console.WriteLine("Training Completed.\n");
        }

        /// <summary>
        /// Compares original SDRs with reconstructed SDRs for each classifier and computes similarity measures.
        /// Returns a dictionary of image name to (HTM similarity, k-NN similarity) as percentage.
        /// </summary>
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

                // Compute similarity measures (Jaccard, Cosine, Hamming) for each reconstruction
                double origVsHtm_Jaccard = MathHelpers.JaccardSimilarityofBinaryArrays(origSdr, htmReconSdr);
                double origVsHtm_Cosine = ComputeCosineSimilarity(origSdr, htmReconSdr);
                double origVsHtm_Hamming = ComputeHammingSimilarity(origSdr, htmReconSdr);
                double htmSimAvg = (origVsHtm_Jaccard + origVsHtm_Cosine + origVsHtm_Hamming) / 3.0 * 100.0;

                double origVsKnn_Jaccard = MathHelpers.JaccardSimilarityofBinaryArrays(origSdr, knnReconSdr);
                double origVsKnn_Cosine = ComputeCosineSimilarity(origSdr, knnReconSdr);
                double origVsKnn_Hamming = ComputeHammingSimilarity(origSdr, knnReconSdr);
                double knnSimAvg = (origVsKnn_Jaccard + origVsKnn_Cosine + origVsKnn_Hamming) / 3.0 * 100.0;

                similarityResults[name] = (htmSimAvg, knnSimAvg);

                // Log similarity results for this image, including each measure and the average percentage
                Console.WriteLine($"\nSimilarity Results for \"{name}\":");
                Console.WriteLine($" HTM - Jaccard: {origVsHtm_Jaccard * 100:F2}%, Cosine: {origVsHtm_Cosine * 100:F2}%, Hamming: {origVsHtm_Hamming * 100:F2}% (Avg: {htmSimAvg:F2}%)");
                Console.WriteLine($" KNN - Jaccard: {origVsKnn_Jaccard * 100:F2}%, Cosine: {origVsKnn_Cosine * 100:F2}%, Hamming: {origVsKnn_Hamming * 100:F2}% (Avg: {knnSimAvg:F2}%)");
            }
            return similarityResults;
        }

        /// <summary>
        /// Utility to read an SDR from a text file (comma-separated 0/1 values) into an int[].
        /// </summary>
        private static int[] ReadSdrFromFile(string path)
        {
            return File.ReadAllText(path).Trim()
                       .Replace("\r", "").Replace("\n", "")  // ensure no line breaks
                       .Split(',')
                       .Where(str => !string.IsNullOrWhiteSpace(str))
                       .Select(str => int.TryParse(str, out int bit) ? bit : 0)
                       .ToArray();
        }

        /// <summary>
        /// Computes cosine similarity between two binary SDRs (treating them as vectors).
        /// Returns a value between 0 and 1 (1 means identical, 0 means completely different).
        /// </summary>
        private static double ComputeCosineSimilarity(int[] sdr1, int[] sdr2)
        {
            if (sdr1.Length != sdr2.Length)
                throw new ArgumentException("SDRs must be the same length for cosine similarity");
            double dot = sdr1.Zip(sdr2, (a, b) => a * b).Sum();
            double magA = Math.Sqrt(sdr1.Sum(a => a * a));
            double magB = Math.Sqrt(sdr2.Sum(b => b * b));
            if (magA == 0 || magB == 0) return 0.0;
            return dot / (magA * magB);
        }

        /// <summary>
        /// Computes Hamming similarity between two binary SDRs.
        /// This is the fraction of bit positions that are the same between the two SDRs (1.0 means identical).
        /// </summary>
        private static double ComputeHammingSimilarity(int[] sdr1, int[] sdr2)
        {
            if (sdr1.Length != sdr2.Length)
                throw new ArgumentException("SDRs must be the same length for Hamming similarity");
            int matchingBits = sdr1.Zip(sdr2, (a, b) => a == b ? 1 : 0).Sum();
            return (double)matchingBits / sdr1.Length;
        }

        /// <summary>
        /// Ensures a directory exists; if it does not, creates it.
        /// </summary>
        private static void EnsureDirectoryExists(string path)
        {
            if (!Directory.Exists(path))
            {
                Directory.CreateDirectory(path);
            }
        }

        /// <summary>
        /// Generates a simple bar graph image comparing HTM vs k-NN similarity scores for each image.
        /// Saves the graph to an output file for visual comparison of classifier performance.
        /// </summary>
        private static void GenerateSimilarityGraph(Dictionary<string, (double htmSim, double knnSim)> similarityResults)
        {
            if (similarityResults == null || similarityResults.Count == 0)
            {
                Console.WriteLine("No similarity results to plot.");
                return;
            }

            string outputFolder = Path.Combine(Environment.CurrentDirectory, "SimilarityPlots_Image_Inputs");
            EnsureDirectoryExists(outputFolder);

            int width = 800;
            int height = 400;
            using Bitmap bmp = new Bitmap(width, height);
            using Graphics g = Graphics.FromImage(bmp);
            g.Clear(Color.White);

            // Determine the maximum similarity percentage to scale bars appropriately
            double maxHtmSim = similarityResults.Values.Max(r => r.htmSim);
            double maxKnnSim = similarityResults.Values.Max(r => r.knnSim);
            double maxSim = Math.Max(maxHtmSim, maxKnnSim);

            // Set up dimensions for bar chart
            int barWidth = width / (2 * similarityResults.Count + 1);
            int padding = 10;
            int baseLineY = height - 50;
            int xPosition = padding;

            // Draw bars for each image's similarities
            foreach (var result in similarityResults)
            {
                string name = result.Key;
                double htmSim = result.Value.htmSim;
                double knnSim = result.Value.knnSim;

                // Bar heights proportional to similarity (normalized by maxSim)
                int htmBarHeight = (int)((htmSim / maxSim) * (height - 60));
                int knnBarHeight = (int)((knnSim / maxSim) * (height - 60));

                // Draw HTM bar (blue) and k-NN bar (green)
                g.FillRectangle(Brushes.Blue, xPosition, baseLineY - htmBarHeight, barWidth, htmBarHeight);
                g.FillRectangle(Brushes.Green, xPosition + barWidth, baseLineY - knnBarHeight, barWidth, knnBarHeight);

                // Label each pair of bars with the image name and classifier labels
                g.DrawString(name, new Font("Arial", 8), Brushes.Black, new PointF(xPosition, baseLineY + 5));
                g.DrawString("HTM", new Font("Arial", 8), Brushes.Blue, new PointF(xPosition, baseLineY - htmBarHeight - 20));
                g.DrawString("KNN", new Font("Arial", 8), Brushes.Green, new PointF(xPosition + barWidth, baseLineY - knnBarHeight - 20));

                xPosition += 2 * barWidth + padding;
            }

            string outputPath = Path.Combine(outputFolder, "SimilarityComparison.png");
            bmp.Save(outputPath);
            Console.WriteLine($"\nSimilarity comparison chart saved to {outputPath}");
        }
    }
}
