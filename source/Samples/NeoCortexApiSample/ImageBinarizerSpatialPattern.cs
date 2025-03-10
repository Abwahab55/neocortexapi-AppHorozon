using NeoCortexApi;
using NeoCortexApi.Entities;
using NeoCortexApi.Utility;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace NeoCortexApiSample
{
    internal class ImageBinarizerSpatialPattern
    {
        public void Run()
        {
            Console.WriteLine($"Starting Experiment: {nameof(ImageBinarizerSpatialPattern)}");

            int numColumns = 32 * 32;  // 1024 columns
            int imageSize = 28;        // 28x28 images
            var colDims = new int[] { 32, 32 };

            HtmConfig cfg = new HtmConfig(new int[] { imageSize, imageSize }, new int[] { numColumns })
            {
                CellsPerColumn = 10,
                InputDimensions = new int[] { imageSize, imageSize },
                NumInputs = imageSize * imageSize,
                ColumnDimensions = colDims,
                MaxBoost = 3.0,
                DutyCyclePeriod = 100,
                MinPctOverlapDutyCycles = 0.10,
                GlobalInhibition = true,
                NumActiveColumnsPerInhArea = 0.05 * numColumns,
                PotentialRadius = (int)(0.15 * imageSize * imageSize),
                LocalAreaDensity = -1,
                ActivationThreshold = 5,
                MaxSynapsesPerSegment = (int)(0.02 * numColumns),
                Random = new ThreadSafeRandom(42),
                StimulusThreshold = 4,
            };

            var sp = RunExperiment(cfg);
            if (sp != null) RunRestructuringExperiment(cfg);
        }

        private string AdaptiveBinarizeImage(string imagePath, int imageSize, string outputName)
        {
            Mat image = Cv2.ImRead(imagePath, ImreadModes.Grayscale);
            Cv2.Resize(image, image, new OpenCvSharp.Size(imageSize, imageSize));

            Mat binaryImage = new Mat();
            Cv2.AdaptiveThreshold(image, binaryImage, 255, AdaptiveThresholdTypes.GaussianC, ThresholdTypes.Binary, 11, 2);

            string outputFolder = Path.Combine(Environment.CurrentDirectory, "BinarizedImages");
            if (!Directory.Exists(outputFolder))
                Directory.CreateDirectory(outputFolder);

            string outputFile = Path.Combine(outputFolder, $"{outputName}.csv");

            using (StreamWriter writer = new StreamWriter(outputFile))
            {
                for (int y = 0; y < 28; y++) // Ensure exactly 28x28 = 784 values
                {
                    string line = string.Join(",", Enumerable.Range(0, 28)
                                            .Select(x => binaryImage.At<byte>(y, x) > 128 ? "1" : "0"));
                    writer.WriteLine(line);
                }
            }

            Console.WriteLine($"Binarized Image Saved: {outputFile}");
            return outputFile;
        }

        private SpatialPooler RunExperiment(HtmConfig cfg)
        {
            Console.WriteLine("Running Experiment...");
            string trainingFolder = Path.Combine(Environment.CurrentDirectory, "Sample");
            var trainingImages = Directory.GetFiles(trainingFolder, "*.png");

            if (trainingImages.Length == 0)
            {
                Console.WriteLine("No images found in 'Sample' folder.");
                return null;
            }

            string sdrFolder = Path.Combine(Environment.CurrentDirectory, "SDR_Values");
            if (!Directory.Exists(sdrFolder))
                Directory.CreateDirectory(sdrFolder);

            foreach (var image in trainingImages)
            {
                Console.WriteLine($"Processing Image: {image}");
                var mem = new Connections(cfg);
                SpatialPooler sp = new SpatialPooler();
                sp.Init(mem);

                string imageName = Path.GetFileNameWithoutExtension(image);
                string inputBinaryImageFile = AdaptiveBinarizeImage(image, 28, imageName);

                try
                {
                    var inputVector = File.ReadAllLines(inputBinaryImageFile)
                                        .SelectMany(line => line.Split(',')
                                        .Select(value => int.TryParse(value, out int num) ? num : 0))
                                        .ToArray();

                    if (inputVector.Length != 784)
                    {
                        Console.WriteLine($"WARNING: Adjusting input vector size from {inputVector.Length} to 784.");
                        Array.Resize(ref inputVector, 784);
                    }

                    int[] activeArray = new int[32 * 32];
                    sp.compute(inputVector, activeArray, true);
                    var activeCols = ArrayUtils.IndexWhere(activeArray, el => el == 1);

                    if (activeCols.Length == 0)
                    {
                        Console.WriteLine($"WARNING: No active SDR columns for {imageName}. Adjust thresholds.");
                    }

                    string sdrFile = Path.Combine(sdrFolder, $"sdr_{imageName}.csv");
                    File.WriteAllLines(sdrFile, activeCols.Select(x => x.ToString()));
                    Console.WriteLine($"SDR values saved in {sdrFile}");
                }
                catch (System.Exception ex)
                {
                    Console.WriteLine($"ERROR: Could not process {imageName}. {ex.Message}");
                }
            }

            return new SpatialPooler();
        }

        private void RunRestructuringExperiment(HtmConfig cfg)
        {
            Console.WriteLine("Running Restructuring Experiment...");
            string trainingFolder = Path.Combine(Environment.CurrentDirectory, "Sample");
            var trainingImages = Directory.GetFiles(trainingFolder, "*.png");

            if (trainingImages.Length == 0)
            {
                Console.WriteLine("No images found for restructuring.");
                return;
            }

            int imgSize = 28;
            int[] activeArray = new int[32 * 32];

            foreach (var image in trainingImages)
            {
                Console.WriteLine($"Processing image: {image}");

                var mem = new Connections(cfg);
                SpatialPooler sp = new SpatialPooler();
                sp.Init(mem);

                string imageName = Path.GetFileNameWithoutExtension(image);
                string inputBinaryImageFile = AdaptiveBinarizeImage(image, imgSize, imageName);

                var inputVector = File.ReadAllLines(inputBinaryImageFile)
                                    .SelectMany(line => line.Split(',')
                                    .Select(value => int.TryParse(value, out int num) ? num : 0))
                                    .ToArray();

                sp.compute(inputVector, activeArray, true);
                var activeCols = ArrayUtils.IndexWhere(activeArray, (el) => el == 1);

                Console.WriteLine($"SDR Output for {imageName}: {string.Join(",", activeCols)}");
            }
        }
    }
}
