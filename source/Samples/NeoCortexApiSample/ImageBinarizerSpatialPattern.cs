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
<<<<<<< HEAD
        public void Run()
        {
            Console.WriteLine($"Starting Experiment: {nameof(ImageBinarizerSpatialPattern)}");

            int numColumns = 32 * 32;  // 1024 columns
            int imageSize = 28;        // 28x28 images
=======
        public string inputPrefix { get; private set; } = "input_";

        public void Run()
<<<<<<< HEAD
=======

>>>>>>> 578a6101a446a7ccad67b4f5976fbeb70c4143d0
        {   //EXPERIMENT OF IMAGE BINARIZATION
            Console.WriteLine($" Starting Experiment: {nameof(ImageBinarizerSpatialPattern)}");
            double minOctOverlapCycles = 1.0;
            double maxBoost = 5.0;
            int numColumns = 32 * 32;
            int imageSize = 28;
>>>>>>> d2b65405fac2dd582dde9c2d76774196e41806f4
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
<<<<<<< HEAD
            if (sp != null) RunRestructuringExperiment(cfg);
=======
            RunRestructuringExperiment(sp);
>>>>>>> d2b65405fac2dd582dde9c2d76774196e41806f4
        }

        private string AdaptiveBinarizeImage(string imagePath, int imageSize, string outputName)
        {
            Mat image = Cv2.ImRead(imagePath, ImreadModes.Grayscale);
            Cv2.Resize(image, image, new OpenCvSharp.Size(imageSize, imageSize));

            Mat binaryImage = new Mat();
            Cv2.AdaptiveThreshold(image, binaryImage, 255, AdaptiveThresholdTypes.GaussianC, ThresholdTypes.Binary, 11, 2);

<<<<<<< HEAD
=======
<<<<<<< Updated upstream
            string outputFolderCsv = Path.Combine(Environment.CurrentDirectory, "BinarizedImages");
            string outputFolderPng = Path.Combine(Environment.CurrentDirectory, "BinarizedImages_PNG");
            Directory.CreateDirectory(outputFolderCsv);
            Directory.CreateDirectory(outputFolderPng);
=======
            // Convert the binarized image to a Numeric CSV format
>>>>>>> d2b65405fac2dd582dde9c2d76774196e41806f4
            string outputFolder = Path.Combine(Environment.CurrentDirectory, "BinarizedImages");
            if (!Directory.Exists(outputFolder))
                Directory.CreateDirectory(outputFolder);
>>>>>>> Stashed changes

            string outputCsvFile = Path.Combine(outputFolderCsv, $"{outputName}.csv");
            string outputPngFile = Path.Combine(outputFolderPng, $"{outputName}.png");

            // Save as CSV (Numbers)
            using (StreamWriter writer = new StreamWriter(outputCsvFile))
            {
<<<<<<< HEAD
                for (int y = 0; y < 28; y++) // Ensure exactly 28x28 = 784 values
                {
                    string line = string.Join(",", Enumerable.Range(0, 28)
                                            .Select(x => binaryImage.At<byte>(y, x) > 128 ? "1" : "0"));
                    writer.WriteLine(line);
                }
            }

            Console.WriteLine($"Binarized Image Saved: {outputFile}");
=======
                var indexer = binaryImage.GetGenericIndexer<byte>();
                for (int i = 0; i < binaryImage.Rows; i++)
                {
                    List<string> rowValues = new List<string>();
                    for (int j = 0; j < binaryImage.Cols; j++)
                    {
                        rowValues.Add(indexer[i, j] > 0 ? "1" : "0");
                    }
                    writer.WriteLine(string.Join(",", rowValues));
                }
            }
<<<<<<< HEAD
=======


            // Save as PNG (Image)
            Cv2.ImWrite(outputPngFile, binaryImage);

            Console.WriteLine($" Binarized Image Saved (CSV): {outputCsvFile}");
            Console.WriteLine($" Binarized Image Saved (PNG): {outputPngFile}");

            return outputCsvFile;
>>>>>>> 578a6101a446a7ccad67b4f5976fbeb70c4143d0
            //SAVED BINARIZED IMAGE AS OUTPUT 
            Console.WriteLine($" Binarized Image Saved: {outputFile}");
>>>>>>> d2b65405fac2dd582dde9c2d76774196e41806f4
            return outputFile;
        }

        private SpatialPooler RunExperiment(HtmConfig cfg)
        {
<<<<<<< HEAD
            Console.WriteLine("Running Experiment...");
            string trainingFolder = Path.Combine(Environment.CurrentDirectory, "Sample");
            var trainingImages = Directory.GetFiles(trainingFolder, "*.png");

=======
            Console.WriteLine("?? Running Experiment...");
            var mem = new Connections(cfg);
            bool isInStableState = false;
            int numColumns = 32 * 32;
<<<<<<< HEAD

            // PATH SPECIFICATION STEPS
=======
>>>>>>> 578a6101a446a7ccad67b4f5976fbeb70c4143d0
            string trainingFolder = Path.Combine(Environment.CurrentDirectory, "Sample");
            var trainingImages = Directory.GetFiles(trainingFolder, $"{inputPrefix}*.png");


<<<<<<< HEAD
            //TRAINING FOLDER
            Console.WriteLine($" Looking for images in: {trainingFolder}");
            var trainingImages = Directory.GetFiles(trainingFolder, "*.png");
=======
>>>>>>> 578a6101a446a7ccad67b4f5976fbeb70c4143d0
>>>>>>> d2b65405fac2dd582dde9c2d76774196e41806f4
            if (trainingImages.Length == 0)
            {
                Console.WriteLine("No images found in 'Sample' folder.");
                return null;
            }
<<<<<<< HEAD
=======
<<<<<<< HEAD
=======

            string sdrFolder = Path.Combine(Environment.CurrentDirectory, "SDR_Values");
            Directory.CreateDirectory(sdrFolder);

            //TRAINING FOLDER
            Console.WriteLine($" Looking for images in: {trainingFolder}");
           
>>>>>>> 578a6101a446a7ccad67b4f5976fbeb70c4143d0
            //IF IMAGES FOUND
            Console.WriteLine($" Found {trainingImages.Length} images in 'Sample' folder.");
            //TEST IMAGE
            string testName = "test_image";

            HomeostaticPlasticityController hpa = new HomeostaticPlasticityController(mem, trainingImages.Length * 50,
                (isStable, numPatterns, actColAvg, seenInputs) =>
                {
                    if (isStable)
                    {
                        Console.WriteLine($"?? STABLE: Patterns={numPatterns}, Inputs={seenInputs}");
                    }
                },
                requiredSimilarityThreshold: 0.975
            );

            SpatialPooler sp = new SpatialPooler(hpa);
            sp.Init(mem, new DistributedMemory() { ColumnDictionary = new InMemoryDistributedDictionary<int, NeoCortexApi.Entities.Column>(1) });

            int[] activeArray = new int[numColumns];
            int maxCycles = 5, currentCycle = 0;
>>>>>>> d2b65405fac2dd582dde9c2d76774196e41806f4

            while (!isInStableState && currentCycle < maxCycles)
            {
<<<<<<< HEAD
                Console.WriteLine($"Processing Image: {image}");
                var mem = new Connections(cfg);
                SpatialPooler sp = new SpatialPooler();
                sp.Init(mem);

                string imageName = Path.GetFileNameWithoutExtension(image);
                string inputBinaryImageFile = AdaptiveBinarizeImage(image, 28, imageName);

                try
=======
                foreach (var image in trainingImages)
>>>>>>> d2b65405fac2dd582dde9c2d76774196e41806f4
                {
                    string inputBinaryImageFile = AdaptiveBinarizeImage(image, 28, Path.GetFileNameWithoutExtension(image));

                    int[] inputVector = ReadCsvIntegersSafe(inputBinaryImageFile);

                    if (inputVector.Length != 784)
                    {
                        Console.WriteLine($"WARNING: Adjusting input vector size from {inputVector.Length} to 784.");
                        Array.Resize(ref inputVector, 784);
                    }

                    int[] activeArray = new int[32 * 32];
                    sp.compute(inputVector, activeArray, true);
<<<<<<< HEAD
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
=======
                    var activeCols = ArrayUtils.IndexWhere(activeArray, (el) => el == 1);

                    string sdrFile = Path.Combine(sdrFolder, $"SDR_{Path.GetFileNameWithoutExtension(image)}.csv");
                    File.WriteAllLines(sdrFile, activeCols.Select(x => x.ToString()));

                    Console.WriteLine($"✅ SDR Values Saved: {sdrFile}");
                    Console.WriteLine($"?? SDR Output: {string.Join(",", activeCols)}");
>>>>>>> d2b65405fac2dd582dde9c2d76774196e41806f4
                }
                currentCycle++;
            }

            return new SpatialPooler();
        }

<<<<<<< HEAD
        private void RunRestructuringExperiment(HtmConfig cfg)
=======
<<<<<<< HEAD
        //
        //RECONSTRUCTION BEGINS(SPATIAL POOLER)
        private void RunRestructuringExperiment(SpatialPooler sp)
>>>>>>> d2b65405fac2dd582dde9c2d76774196e41806f4
        {
            Console.WriteLine("Running Restructuring Experiment...");
            string trainingFolder = Path.Combine(Environment.CurrentDirectory, "Sample");
            var trainingImages = Directory.GetFiles(trainingFolder, "*.png");
<<<<<<< HEAD

            if (trainingImages.Length == 0)
            {
                Console.WriteLine("No images found for restructuring.");
                return;
            }

=======
=======
        //RECONSTRUCTION BEGINS(SPATIAL POOLER)
        private void RunRestructuringExperiment(SpatialPooler sp)
        {

            Console.WriteLine("?? Running Restructuring Experiment...");
            string trainingFolder = Path.Combine(Environment.CurrentDirectory, "Sample");
            var trainingImages = Directory.GetFiles(trainingFolder, $"{inputPrefix}*.png");


>>>>>>> 578a6101a446a7ccad67b4f5976fbeb70c4143d0
            if (trainingImages.Length == 0)
            {
                Console.WriteLine("?? No images found for restructuring.");
                return;
            }
<<<<<<< HEAD
            //PUTTING IMAGE SIZE AS REQUIRED
=======

>>>>>>> 578a6101a446a7ccad67b4f5976fbeb70c4143d0
>>>>>>> d2b65405fac2dd582dde9c2d76774196e41806f4
            int imgSize = 28;
            int[] activeArray = new int[32 * 32];

            foreach (var image in trainingImages)
            {
<<<<<<< HEAD
                Console.WriteLine($"Processing image: {image}");

                var mem = new Connections(cfg);
                SpatialPooler sp = new SpatialPooler();
                sp.Init(mem);

                string imageName = Path.GetFileNameWithoutExtension(image);
                string inputBinaryImageFile = AdaptiveBinarizeImage(image, imgSize, imageName);
=======
                Console.WriteLine($"?? Processing Image: {image}");
                string inputBinaryImageFile = AdaptiveBinarizeImage(image, imgSize, Path.GetFileNameWithoutExtension(image));
>>>>>>> d2b65405fac2dd582dde9c2d76774196e41806f4

                int[] inputVector = ReadCsvIntegersSafe(inputBinaryImageFile);

                sp.compute(inputVector, activeArray, true);
                var activeCols = ArrayUtils.IndexWhere(activeArray, (el) => el == 1);
<<<<<<< HEAD

                Console.WriteLine($"SDR Output for {imageName}: {string.Join(",", activeCols)}");
=======
<<<<<<< HEAD
=======

            }
        }

        private int[] ReadCsvIntegersSafe(string filePath)
        {
            try
            {
                List<int> intList = new List<int>();

                using (var reader = new StreamReader(filePath))
                {
                    while (!reader.EndOfStream)
                    {
                        string line = reader.ReadLine();
                        if (!string.IsNullOrWhiteSpace(line))
                        {
                            intList.AddRange(line.Split(',')
                                .Where(s => int.TryParse(s, out _))
                                .Select(int.Parse));
                        }
                    }
                }

                return intList.ToArray();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Error reading CSV file '{filePath}': {ex.Message}");
                return new int[0];  // Return an empty array to avoid crashes
>>>>>>> 578a6101a446a7ccad67b4f5976fbeb70c4143d0
                //SDR OUTPUT FOR IMAGES
                Console.WriteLine($"📌 SDR Output for {imageName}: {string.Join(",", activeCols)}");
>>>>>>> d2b65405fac2dd582dde9c2d76774196e41806f4
            }
        }
    }
}