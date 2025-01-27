using NeoCortex;
using NeoCortexApi.Entities;
using NeoCortexApi.Utility;
using NeoCortexApi;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace NeoCortexApiSample
{
    internal class ImageBinarizerSpatialPattern
    {
        public string inputPrefix { get; private set; }

        /// <summary>
        /// Implements an experiment that demonstrates how to learn spatial patterns.
        /// SP will learn every presented Image input in multiple iterations.
        /// </summary>
        public void Run()
        {
            Console.WriteLine($"Hello NeocortexApi! Experiment {nameof(ImageBinarizerSpatialPattern)}");

            double minOctOverlapCycles = 1.0;
            double maxBoost = 5.0;
            // We will build a slice of the cortex with the given number of mini-columns
            int numColumns = 64 * 64;
            // The Size of the Image Height and width is 28 pixel
            int imageSize = 28;
            var colDims = new int[] { 64, 64 };

            // This is a set of configuration parameters used in the experiment.
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
                NumActiveColumnsPerInhArea = 0.02 * numColumns,
                PotentialRadius = (int)(0.15 * imageSize * imageSize),
                LocalAreaDensity = -1,
                ActivationThreshold = 10,
                MaxSynapsesPerSegment = (int)(0.01 * numColumns),
                Random = new ThreadSafeRandom(42),
                StimulusThreshold = 10,
            };

            //Runnig the Experiment
            var sp = RunExperiment(cfg, inputPrefix);
            //Runing the Reconstruction Method Experiment
            RunRustructuringExperiment(sp);

        }

        /// <summary>
        /// Implements the experiment.
        /// </summary>
        /// <param name="cfg"></param>
        /// <param name="inputPrefix"> The name of the images</param>
        /// <returns>The trained bersion of the SP.</returns>
        private SpatialPooler RunExperiment(HtmConfig cfg, string inputPrefix)
        {

            var mem = new Connections(cfg);

            bool isInStableState = false;

            int numColumns = 64 * 64;
            //Accessing the Image Folder form the Cureent Directory
            //string trainingFolder = $"Sample";
            string trainingFolder = Path.Combine(Environment.CurrentDirectory, "Sample");

            //Accessing the Image Folder form the Cureent Directory Foldfer
            var trainingImages = Directory.GetFiles(trainingFolder, $"{inputPrefix}*.png");
            //Image Size
            int imageSize = 28;
            //Folder Name in the Directorty 
            string testName = "test_image";

            HomeostaticPlasticityController hpa = new HomeostaticPlasticityController(mem, trainingImages.Length * 50, (isStable, numPatterns, actColAvg, seenInputs) =>
            {
                // Event should only be fired when entering the stable state.
                if (isStable)
                {
                    isInStableState = true;
                    Debug.WriteLine($"Entered STABLE state: Patterns: {numPatterns}, Inputs: {seenInputs}, iteration: {seenInputs / numPatterns}");
                }
                else
                {
                    isInStableState = false;
                    Debug.WriteLine($"INSTABLE STATE");
                }
                // Ideal SP should never enter unstable state after stable state.
                Debug.WriteLine($"Entered STABLE state: Patterns: {numPatterns}, Inputs: {seenInputs}, iteration: {seenInputs / numPatterns}");
            }, requiredSimilarityThreshold: 0.975);

            // It creates the instance of Spatial Pooler Multithreaded version.
            SpatialPooler sp = new SpatialPooler(hpa);

            //Initializing the Spatial Pooler Algorithm for SDR 
            sp.Init(mem, new DistributedMemory() { ColumnDictionary = new InMemoryDistributedDictionary<int, NeoCortexApi.Entities.Column>(1) });

            //Image Size
            int imgSize = 28;
            int[] activeArray = new int[numColumns];

            int numStableCycles = 0;
            // Runnig the Traning Cycle for 5 times
            int maxCycles = 5;
            int currentCycle = 0;

            while (!isInStableState && currentCycle < maxCycles)
            {
                foreach (var Image in trainingImages)
                {
                    //Binarizing the Images before taking Inputs for the Sp
                    string inputBinaryImageFile = NeoCortexUtils.BinarizeImage($"{Image}", imgSize, testName);

                    // Read Binarized and Encoded input csv file into array
                    int[] inputVector = NeoCortexUtils.ReadCsvIntegers(inputBinaryImageFile).ToArray();

                    int[] oldArray = new int[activeArray.Length];
                    List<double[,]> overlapArrays = new List<double[,]>();
                    List<double[,]> bostArrays = new List<double[,]>();

                    sp.compute(inputVector, activeArray, true);
                    //Getting the Active Columns
                    var activeCols = ArrayUtils.IndexWhere(activeArray, (el) => el == 1);

                    Debug.WriteLine($"'Cycle: {currentCycle} - Image-Input: {Image}'");
                    Debug.WriteLine($"INPUT :{Helpers.StringifyVector(inputVector)}");
                    Debug.WriteLine($"SDR:{Helpers.StringifyVector(activeCols)}\n");
                }

                currentCycle++;

                // Check if the desired number of cycles is reached
                if (currentCycle >= maxCycles)
                    break;

                // Increment numStableCycles only when it's in a stable state
                if (isInStableState)
                    numStableCycles++;
            }

            return sp;
        }
        /// <summary>
        /// Runs the restructuring experiment using the provided spatial pooler. 
        /// This method iterates through a set of training images, computes spatial pooling, 
        /// reconstructs permanence values, and generates heatmaps and similarity graphs based on the results.
        /// </summary>
        /// <param name="sp">The spatial pooler to use for the experiment.</param>
        private void RunRustructuringExperiment(SpatialPooler sp)
        {
            // Ensure the training folder is correct
            string trainingFolder = Path.Combine(Environment.CurrentDirectory, "Sample");
            var trainingImages = Directory.GetFiles(trainingFolder, $"{inputPrefix}*.png");

            if (trainingImages.Length == 0)
            {
                Console.WriteLine("No images found in the specified folder with the given prefix.");
                return;
            }

            int imgSize = 28;
            string testName = "test_image";
            int[] activeArray = new int[64 * 64];
            List<List<double>> heatmapData = new List<List<double>>();
            List<int[]> binarizedInputs = new List<int[]>();
            List<int[]> normalizedPermanence = new List<int[]>();
            List<double[]> similarityList = new List<double[]>();

            int i = 1; // Image counter for unique file naming
            foreach (var image in trainingImages)
            {
                Debug.WriteLine($"Processing image: {image}");

                // Binarize the image
                string inputBinaryImageFile = NeoCortexUtils.BinarizeImage(image, imgSize, testName);
                int[] inputVector = NeoCortexUtils.ReadCsvIntegers(inputBinaryImageFile).ToArray();


                sp.compute(inputVector, activeArray, true);
                var activeCols = ArrayUtils.IndexWhere(activeArray, (el) => el == 1);

                Dictionary<int, double> reconstructedPermanence = sp.Reconstruct(activeCols);

                int maxInput = inputVector.Length;

                // Process reconstructed permanence values
                Dictionary<int, double> allPermanenceDictionary = new Dictionary<int, double>();
                foreach (var kvp in reconstructedPermanence)
                {
                    allPermanenceDictionary[kvp.Key] = kvp.Value;
                }

                for (int inputIndex = 0; inputIndex < maxInput; inputIndex++)
                {
                    if (!reconstructedPermanence.ContainsKey(inputIndex))
                    {
                        allPermanenceDictionary[inputIndex] = 0.0;
                    }
                }

                var sortedAllPermanenceDictionary = allPermanenceDictionary.OrderBy(kvp => kvp.Key);
                List<double> permanenceValuesList = sortedAllPermanenceDictionary.Select(kvp => kvp.Value).ToList();

                heatmapData.Add(permanenceValuesList);
                binarizedInputs.Add(inputVector);

                // Normalize permanence
                double thresholdValue = 30.5;
                List<int> normalizePermanenceList = Helpers.ThresholdingProbabilities(permanenceValuesList, thresholdValue);
                normalizedPermanence.Add(normalizePermanenceList.ToArray());

                // Calculate similarity
                double similarity = MathHelpers.JaccardSimilarityofBinaryArrays(inputVector, normalizePermanenceList.ToArray());
                similarityList.Add(new double[] { similarity });

                // Save heatmap for the current image
                string folderPath = Path.Combine(Environment.CurrentDirectory, "1DHeatMap_Image_Inputs");
                if (!Directory.Exists(folderPath))
                {
                    Directory.CreateDirectory(folderPath);
                }
                string heatmapFilePath = Path.Combine(folderPath, $"heatmap_{i}.png");
                Debug.WriteLine($"Heatmap saved for image {i}: {heatmapFilePath}");

                i++;
            }

            // Generate a combined similarity plot for all images
            string similarityPlotPath = Path.Combine(Environment.CurrentDirectory, "SimilarityPlots_Image_Inputs", "combined_similarity_plot.png");
            List<double> combinedSimilarities = similarityList.SelectMany(similarity => similarity).ToList();
            NeoCortexUtils.DrawCombinedSimilarityPlot(combinedSimilarities, similarityPlotPath, 1000, 850);

            Debug.WriteLine($"Combined similarity plot saved: {similarityPlotPath}");
        }

        // <summary>
        /// Draws a combined similarity plot based on the provided list of arrays containing similarity values.
        /// The combined similarity plot is generated by combining all similarity values from the list of arrays,
        /// creating a single list of similarities, and then drawing the plot.
        /// </summary>
        /// <param name="similaritiesList">List of arrays containing similarity values.</param>
        public static void DrawSimilarityPlots(List<double[]> similaritiesList)
        {
            // Combine all similarities from the list of arrays

            List<double> combinedSimilarities = new List<double>();
            foreach (var similarities in similaritiesList)

            {
                combinedSimilarities.AddRange(similarities);
            }

            // Define the folder path based on the current directory

            string folderPath = Path.Combine(Environment.CurrentDirectory, "SimilarityPlots_Image_Inputs");


            // Create the folder if it doesn't exist

            if (!Directory.Exists(folderPath))
            {
                Directory.CreateDirectory(folderPath);
            }

            // Define the file name
            string fileName = "combined_similarity_plot_Image_Inputs.png";

            // Define the file path with the folder path and file name

            string filePath = Path.Combine(folderPath, fileName);

            // Draw the combined similarity plot
            NeoCortexUtils.DrawCombinedSimilarityPlot(combinedSimilarities, filePath, 1000, 850);

            Debug.WriteLine($"Combined similarity plot generated and saved successfully.");

        }
    }
}