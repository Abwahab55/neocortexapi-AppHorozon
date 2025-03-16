using System;
using System.IO;
using System.Linq;
using NeoCortexApi.Classifiers;
using NeoCortexApi.Entities;
using NeoCortexApiSample;

class Program
{
    static void Main(string[] args)
    {
        Console.WriteLine("Starting Image Processing Pipeline...");

        string trainingFolder = Path.Combine(Environment.CurrentDirectory, "Sample");
        string sdrFolder = Path.Combine(Environment.CurrentDirectory, "SDR_Values");
        string outputFolder = Path.Combine(Environment.CurrentDirectory, "ReconstructedImages");

        if (!Directory.Exists(trainingFolder))
        {
            Console.WriteLine($"Error: Training folder '{trainingFolder}' does not exist.");
            return;
        }

        var imageFiles = Directory.GetFiles(trainingFolder, "*.png");
        if (imageFiles.Length == 0)
        {
            Console.WriteLine($"Error: No PNG images found in '{trainingFolder}'.");
            return;
        }

        Console.WriteLine("Running Image Binarization...");
        var binarizer = new ImageBinarizerSpatialPattern(trainingFolder);
        binarizer.Run();
        Console.WriteLine("Image Binarization Completed.");

        if (!Directory.Exists("SDR_Values"))
            Directory.CreateDirectory("SDR_Values");

        var sdrFiles = Directory.GetFiles("SDR_Values", "*.txt");
        if (sdrFiles.Length == 0)
        {
            Console.WriteLine($"Error: No SDR files found in 'SDR_Values'. Exiting...");
            return;
        }

        Console.WriteLine("Initializing Classifiers...");
        IClassifier<int[], string> htmClassifier = new HtmImageClassifier();
        IClassifier<int[], string> knnClassifier = new KnnImageClassifier();

        TrainClassifier(htmClassifier, "SDR_Values", isHtm: true);
        TrainClassifier(knnClassifier, "SDR_Values", isHtm: false);

        RunPredictions(htmClassifier, "SDR_Values", "HTM");
        RunPredictions(knnClassifier, "SDR_Values", "KNN");

        Console.WriteLine("Running HTM Image Reconstruction...");
        HtmImageReconstructor htmReconstructor = new HtmImageReconstructor();
        htmReconstructor.RunReconstruction("SDR_Values", "ReconstructedImages");

        Console.WriteLine("Running KNN Image Reconstruction...");
        KnnImageReconstructor knnReconstructor = new KnnImageReconstructor();
        knnReconstructor.RunReconstruction("SDR_Values", "ReconstructedImages");

        Console.WriteLine("Processing Pipeline Completed.");
    }

    private static void TrainClassifier(IClassifier<int[], string> classifier, string sdrFolder, bool isHtm)
    {
        Console.WriteLine($"Training Classifier: {classifier.GetType().Name}");

        var sdrFiles = Directory.GetFiles(sdrFolder, "*.txt")
                                .OrderBy(x => x)
                                .ToList();

        int cycles = isHtm ? 20 : 1; // increased cycles to help Temporal Memory learn

        for (int cycle = 0; cycle < cycles; cycle++)
        {
            foreach (var sdrFile in sdrFiles)
            {
                string fileName = Path.GetFileNameWithoutExtension(sdrFile);
                int[] sdr = File.ReadAllText(sdrFile)
                                .Trim()
                                .Split(',')
                                .Select(int.Parse)
                                .ToArray();

                classifier.Learn(sdr, new Cell[sdr.Length]);
                Console.WriteLine($"Trained on {fileName} with {sdr.Length} bits (Cycle {cycle + 1}/{cycles}).");
            }
        }
        Console.WriteLine("Training Completed.");
    }

    private static void RunPredictions(IClassifier<int[], string> classifier, string sdrFolder, string method)
    {
        Console.WriteLine($"Running Predictions using {method} Classifier...");

        var sdrFiles = Directory.GetFiles(sdrFolder, "*.txt").OrderBy(x => x).ToList();

        foreach (var sdrFile in sdrFiles)
        {
            string item = Path.GetFileNameWithoutExtension(sdrFile).Replace("input_", "");

            int[] inputSDR = File.ReadAllText(sdrFile)
                                 .Trim()
                                 .Split(',')
                                 .Select(str => int.TryParse(str, out int num) ? num : 0)
                                 .ToArray();

            var predictedSDRs = classifier.GetPredictedInputValues(inputSDR, 3);

            if (predictedSDRs.Count == 0)
            {
                Console.WriteLine($"No predictions for {item}. The model did not return predictive SDRs.");
                continue;
            }

            Console.WriteLine($"Prediction Results for {item} using {method}:");
            foreach (var prediction in predictedSDRs)
            {
                Console.WriteLine($"Similarity: {prediction.Similarity * 100:F2}%");
            }
            Console.WriteLine();
        }
    }

    private static int[] NormalizeSdr(int[] sdr)
    {
        int activeBits = (int)(sdr.Length * 0.3);
        var sortedIndices = sdr
            .Select((value, index) => new { Value = value, Index = index })
            .OrderByDescending(x => x.Value)
            .Take(activeBits)
            .Select(x => x.Index)
            .ToArray();

        int[] normalizedSdr = new int[sdr.Length];
        foreach (int index in sortedIndices)
            normalizedSdr[index] = 1;

        return normalizedSdr;
    }
}
