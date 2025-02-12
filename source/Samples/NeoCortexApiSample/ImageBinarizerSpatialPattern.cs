using NeoCortex;
using NeoCortexApi.Entities;
using NeoCortexApi.Utility;
using NeoCortexApi;
using System;
using System.IO;
using System.Linq;
using OpenCvSharp;

namespace NeoCortexApiSample
{
    internal class ImageBinarizerSpatialPattern
    {
        public string inputPrefix { get; private set; } = "";

        public void Run()
        {
            Console.WriteLine($" Starting Experiment: {nameof(ImageBinarizerSpatialPattern)}");

            double minOctOverlapCycles = 1.0;
            double maxBoost = 5.0;
            int numColumns = 32 * 32;
            int imageSize = 28;
            var colDims = new int[] { 32, 32 };

            HtmConfig cfg = new HtmConfig(new int[] { imageSize, imageSize }, new int[] { numColumns })
            {
                CellsPerColumn = 10,
                InputDimensions = new int[] { imageSize, imageSize },
                NumInputs = imageSize * imageSize,
                ColumnDimensions = colDims,
                MaxBoost = maxBoost,
                DutyCyclePeriod = 100,
                MinPctOverlapDutyCycles = minOctOverlapCycles,
                GlobalInhibition = false,
                NumActiveColumnsPerInhArea = 0.03 * numColumns,
                PotentialRadius = (int)(0.2 * imageSize * imageSize),
                LocalAreaDensity = -1,
                ActivationThreshold = 8,
                MaxSynapsesPerSegment = (int)(0.015 * numColumns),
                Random = new ThreadSafeRandom(42),
                StimulusThreshold = 8,
            };

            var sp = RunExperiment(cfg);
            if (sp != null) RunRestructuringExperiment(sp);
        }

        private string AdaptiveBinarizeImage(string imagePath, int imageSize, string outputName)
        {
            Mat image = Cv2.ImRead(imagePath, ImreadModes.Grayscale);
            Cv2.Resize(image, image, new OpenCvSharp.Size(imageSize, imageSize));

            Mat binaryImage = new Mat();
            Cv2.AdaptiveThreshold(image, binaryImage, 255, AdaptiveThresholdTypes.GaussianC, ThresholdTypes.Binary, 11, 2);

            // Convert the binarized image to a numeric CSV format
            string outputFolder = Path.Combine(Environment.CurrentDirectory, "BinarizedImages");
            if (!Directory.Exists(outputFolder))
                Directory.CreateDirectory(outputFolder);

            string outputFile = Path.Combine(outputFolder, $"{outputName}.csv");

            using (StreamWriter writer = new StreamWriter(outputFile))
            {
                for (int y = 0; y < binaryImage.Rows; y++)
                {
                    string line = string.Join(",", Enumerable.Range(0, binaryImage.Cols)
                                            .Select(x => binaryImage.At<byte>(y, x) > 128 ? "1" : "0"));
                    writer.WriteLine(line);
                }
            }

            Console.WriteLine($" Binarized Image Saved: {outputFile}");
            return outputFile;
        }

        private SpatialPooler RunExperiment(HtmConfig cfg)
        {
            Console.WriteLine(" Running Experiment...");
            var mem = new Connections(cfg);
            bool isInStableState = false;
            int numColumns = 32 * 32;
            string trainingFolder = Path.Combine(Environment.CurrentDirectory, "Sample");

            Console.WriteLine($" Looking for images in: {trainingFolder}");
            var trainingImages = Directory.GetFiles(trainingFolder, "*.png");
            if (trainingImages.Length == 0)
            {
                Console.WriteLine(" No images found in the 'Sample' folder.");
                return null;
            }

            Console.WriteLine($" Found {trainingImages.Length} images in 'Sample' folder.");

            string testName = "test_image";

            HomeostaticPlasticityController hpa = new HomeostaticPlasticityController(mem, trainingImages.Length * 50,
                (isStable, numPatterns, actColAvg, seenInputs) =>
                {
                    if (isStable)
                    {
                        Console.WriteLine($"STABLE: Patterns={numPatterns}, Inputs={seenInputs}");
                    }
                },
                requiredSimilarityThreshold: 0.975
            );

            SpatialPooler sp = new SpatialPooler(hpa);
            sp.Init(mem, new DistributedMemory() { ColumnDictionary = new InMemoryDistributedDictionary<int, NeoCortexApi.Entities.Column>(1) });

            int[] activeArray = new int[numColumns];

            string sdrFolder = Path.Combine(Environment.CurrentDirectory, "SDR_Values");
            if (!Directory.Exists(sdrFolder))
                Directory.CreateDirectory(sdrFolder);

            foreach (var image in trainingImages)
            {
                string imageName = Path.GetFileNameWithoutExtension(image);
                string inputBinaryImageFile = AdaptiveBinarizeImage(image, 28, imageName);

                try
                {
                    var inputVector = File.ReadAllLines(inputBinaryImageFile)
                                        .SelectMany(line => line.Split(',')
                                        .Select(value => int.TryParse(value, out int num) ? num : 0))
                                        .ToArray();

                    sp.compute(inputVector, activeArray, true);
                    var activeCols = ArrayUtils.IndexWhere(activeArray, (el) => el == 1);

                    string sdrFile = Path.Combine(sdrFolder, $"sdr_{imageName}.csv");
                    File.WriteAllLines(sdrFile, activeCols.Select(x => x.ToString()));

                    Console.WriteLine($" SDR values saved in {sdrFile}");
                    Console.WriteLine($" SDR for {imageName}: {string.Join(",", activeCols)}");

                    if (activeCols.Length == 0)
                    {
                        Console.WriteLine($" WARNING: SDR for {imageName} is empty! Check binarization.");
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($" ERROR: Could not process {imageName}. {ex.Message}");
                }
            }

            return sp;
        }

        private void RunRestructuringExperiment(SpatialPooler sp)
        {
            Console.WriteLine(" Running Restructuring Experiment...");
            string trainingFolder = Path.Combine(Environment.CurrentDirectory, "Sample");
            var trainingImages = Directory.GetFiles(trainingFolder, "*.png");

            if (trainingImages.Length == 0)
            {
                Console.WriteLine(" No images found for restructuring.");
                return;
            }

            int imgSize = 28;
            int[] activeArray = new int[32 * 32];

            foreach (var image in trainingImages)
            {
                Console.WriteLine($" Processing image: {image}");
                string imageName = Path.GetFileNameWithoutExtension(image);
                string inputBinaryImageFile = AdaptiveBinarizeImage(image, imgSize, imageName);

                var inputVector = File.ReadAllLines(inputBinaryImageFile)
                                    .SelectMany(line => line.Split(',')
                                    .Select(value => int.TryParse(value, out int num) ? num : 0))
                                    .ToArray();

                sp.compute(inputVector, activeArray, true);
                var activeCols = ArrayUtils.IndexWhere(activeArray, (el) => el == 1);

                Console.WriteLine($" SDR Output for {imageName}: {string.Join(",", activeCols)}");
            }
        }
    }
}
