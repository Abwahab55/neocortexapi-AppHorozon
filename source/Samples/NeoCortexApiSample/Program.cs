using System;
using System.IO;
using System.Linq;
using NeoCortexApi.Classifiers;
using NeoCortexApi.Entities;
using NeoCortexApi.Utility;
using NeoCortexApiSample;

class Program
{
    static void Main(string[] args)
    {
        Console.WriteLine("Starting Image Processing Pipeline...");

        string trainingFolder = Path.Combine(Environment.CurrentDirectory, "Sample");
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

        TrainClassifier(htmClassifier, sdrFolder, isHtm: true);
        TrainClassifier(knnClassifier, sdrFolder, isHtm: false);

        Console.WriteLine("Running Image Reconstruction via Classifiers...");
        var htmReconstructor = new HtmImageReconstructor();
        htmReconstructor.RunReconstruction(sdrFolder, outputFolder, reconstructedSdrFolder, htmClassifier);
        var knnReconstructor = new KnnImageReconstructor();
        knnReconstructor.RunReconstruction(sdrFolder, outputFolder, reconstructedSdrFolder, knnClassifier);

        Console.WriteLine("Computing Similarity between Original and Reconstructed SDRs...");
        CompareOriginalAndReconstructedSDRs(sdrFolder, reconstructedSdrFolder);

        Console.WriteLine("Processing Pipeline Completed.");
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

        Console.WriteLine("Training Completed.\n");
    }

    private static void CompareOriginalAndReconstructedSDRs(string sdrFolder, string reconstructedSdrFolder)
    {
        var originalFiles = Directory.GetFiles(sdrFolder, "*.txt").OrderBy(x => x).ToList();

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

            double htmSim = ComputeHybridSimilarity(origSdr, htmReconSdr);
            double knnSim = ComputeHybridSimilarity(origSdr, knnReconSdr);

            Console.WriteLine($"Similarity Results for {name}:");
            Console.WriteLine($" HTM Similarity: {htmSim:0.00}%");
            Console.WriteLine($" KNN Similarity: {knnSim:0.00}%\n");
        }
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
        double hammingSim = ComputeHammingSimilarity(sdr1, sdr2);

        return (jaccardSim + hammingSim) / 2.0;
    }

    private static double ComputeHammingSimilarity(int[] sdr1, int[] sdr2)
    {
        if (sdr1.Length != sdr2.Length)
            throw new ArgumentException("SDRs must be the same length");

        int matchingBits = sdr1.Zip(sdr2, (a, b) => a == b ? 1 : 0).Sum();
        return 100.0 * matchingBits / sdr1.Length;
    }

    private static void EnsureDirectoryExists(string path)
    {
        if (!Directory.Exists(path))
            Directory.CreateDirectory(path);
    }
}
