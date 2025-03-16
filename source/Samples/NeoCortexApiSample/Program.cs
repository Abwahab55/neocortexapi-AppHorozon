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

        if (!Directory.Exists(sdrFolder))
            Directory.CreateDirectory(sdrFolder);

        var sdrFiles = Directory.GetFiles(sdrFolder, "*.txt");
        if (sdrFiles.Length == 0)
        {
            Console.WriteLine($"Error: No SDR files found in '{sdrFolder}'. Exiting...");
            return;
        }

        Console.WriteLine("Initializing Classifiers...");
        IClassifier<int[], string> htmClassifier = new HtmImageClassifier();
        IClassifier<int[], string> knnClassifier = new KnnImageClassifier();

        TrainClassifier(htmClassifier, sdrFolder, isHtm: true);
        TrainClassifier(knnClassifier, sdrFolder, isHtm: false);

        RunPredictions(htmClassifier, sdrFolder, "HTM");
        RunPredictions(knnClassifier, sdrFolder, "KNN");

        Console.WriteLine("Running HTM Image Reconstruction...");
        HtmImageReconstructor htmReconstructor = new HtmImageReconstructor();
        htmReconstructor.RunReconstruction(sdrFolder, outputFolder);

        Console.WriteLine("Running KNN Image Reconstruction...");
        KnnImageReconstructor knnReconstructor = new KnnImageReconstructor();
        knnReconstructor.RunReconstruction(sdrFolder, outputFolder);

        Console.WriteLine("Processing Pipeline Completed.");
    }

    private static void TrainClassifier(IClassifier<int[], string> classifier, string sdrFolder, bool isHtm)
    {
        Console.WriteLine($"Training Classifier: {classifier.GetType().Name}");

        var sdrFiles = Directory.GetFiles(sdrFolder, "*.txt").OrderBy(x => x).ToList();

        int trainingCycles = isHtm ? 10 : 1;

        for (int cycle = 0; cycle < trainingCycles; cycle++)
        {
            foreach (var sdrFile in sdrFiles)
            {
                string fileName = Path.GetFileNameWithoutExtension(sdrFile);

                int[] sdr = File.ReadAllText(sdrFile)
                                .Trim()
                                .Split(',')
                                .Select(str => int.TryParse(str, out int num) ? num : 0)
                                .ToArray();

                sdr = NormalizeSdr(sdr);
                classifier.Learn(sdr, new Cell[sdr.Length]);

                Console.WriteLine($"Trained on {fileName} with {sdr.Length} bits (Cycle {cycle + 1}/{trainingCycles}).");
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
        var sortedIndices = sdr.Select((v, i) => new { v, i })
                               .OrderByDescending(x => x.v)
                               .Take(activeBits)
                               .Select(x => x.i);

        int[] normalizedSdr = new int[sdr.Length];
        foreach (int index in sortedIndices)
            normalizedSdr[index] = 1;

        return normalizedSdr;
    }
}
