using NeoCortex;
using NeoCortexApi.Entities;
using NeoCortexApi.Utility;
using NeoCortexApi;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using OpenCvSharp;

namespace NeoCortexApiSample
{
    internal class ImageBinarizerSpatialPattern
    {
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
            //RUNEXPERIMENT
            var sp = RunExperiment(cfg);
            RunRestructuringExperiment(sp);
        }

        private string AdaptiveBinarizeImage(string imagePath, int imageSize, string outputName)
        {
            Mat image = Cv2.ImRead(imagePath, ImreadModes.Grayscale);
            Cv2.Resize(image, image, new OpenCvSharp.Size(imageSize, imageSize));

            Mat binaryImage = new Mat();
            Cv2.AdaptiveThreshold(image, binaryImage, 255, AdaptiveThresholdTypes.GaussianC, ThresholdTypes.Binary, 11, 2);

<<<<<<< Updated upstream
            string outputFolderCsv = Path.Combine(Environment.CurrentDirectory, "BinarizedImages");
            string outputFolderPng = Path.Combine(Environment.CurrentDirectory, "BinarizedImages_PNG");
            Directory.CreateDirectory(outputFolderCsv);
            Directory.CreateDirectory(outputFolderPng);
=======
            // Convert the binarized image to a Numeric CSV format
            string outputFolder = Path.Combine(Environment.CurrentDirectory, "BinarizedImages");
            if (!Directory.Exists(outputFolder))
                Directory.CreateDirectory(outputFolder);
>>>>>>> Stashed changes

            string outputCsvFile = Path.Combine(outputFolderCsv, $"{outputName}.csv");
            string outputPngFile = Path.Combine(outputFolderPng, $"{outputName}.png");

            // Save as CSV (Numbers)
            using (StreamWriter writer = new StreamWriter(outputCsvFile))
            {
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
            return outputFile;
        }
        //SPATIAL POOLER EXPERIMENT
        private SpatialPooler RunExperiment(HtmConfig cfg)
        {
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
            if (trainingImages.Length == 0)
            {
                //IF IMAGES NOT FOUND
                Console.WriteLine(" No images found in the 'Sample' folder.");
                return null;
            }
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

            while (!isInStableState && currentCycle < maxCycles)
            {
                foreach (var image in trainingImages)
                {
                    string inputBinaryImageFile = AdaptiveBinarizeImage(image, 28, Path.GetFileNameWithoutExtension(image));

                    int[] inputVector = ReadCsvIntegersSafe(inputBinaryImageFile);

                    sp.compute(inputVector, activeArray, true);
                    var activeCols = ArrayUtils.IndexWhere(activeArray, (el) => el == 1);

                    string sdrFile = Path.Combine(sdrFolder, $"SDR_{Path.GetFileNameWithoutExtension(image)}.csv");
                    File.WriteAllLines(sdrFile, activeCols.Select(x => x.ToString()));

                    Console.WriteLine($"✅ SDR Values Saved: {sdrFile}");
                    Console.WriteLine($"?? SDR Output: {string.Join(",", activeCols)}");
                }
                currentCycle++;
            }

            return sp;
        }

<<<<<<< HEAD
        //
        //RECONSTRUCTION BEGINS(SPATIAL POOLER)
        private void RunRestructuringExperiment(SpatialPooler sp)
        {
            
            Console.WriteLine(" Running Restructuring Experiment...");
            string trainingFolder = Path.Combine(Environment.CurrentDirectory, "Sample");
            var trainingImages = Directory.GetFiles(trainingFolder, "*.png");
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
            int imgSize = 28;
            int[] activeArray = new int[32 * 32];

            foreach (var image in trainingImages)
            {
                Console.WriteLine($"?? Processing Image: {image}");
                string inputBinaryImageFile = AdaptiveBinarizeImage(image, imgSize, Path.GetFileNameWithoutExtension(image));

                int[] inputVector = ReadCsvIntegersSafe(inputBinaryImageFile);

                sp.compute(inputVector, activeArray, true);
                var activeCols = ArrayUtils.IndexWhere(activeArray, (el) => el == 1);
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
            }
        }
    }
}