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

        var knnReconstructor = new KnnImageReconstructor(64, 64, k: 5);
        knnReconstructor.RunReconstruction(sdrFolder, outputFolder, reconstructedSdrFolder, knnClassifier);

        Console.WriteLine("Computing Similarity between Original and Reconstructed SDRs...");
        CompareOriginalAndReconstructedSDRs(sdrFolder, reconstructedSdrFolder);

        Console.WriteLine("Processing Pipeline Completed.");
    }

    private static void TrainClassifier(IClassifier<int[], string> classifier, string sdrFolder, bool isHtm)
    {
        Console.WriteLine($"Training Classifier: {classifier.GetType().Name}");
        var sdrFiles = Directory.GetFiles(sdrFolder, "*.txt")
            .Where(file => !file.EndsWith("_binarized.txt")) // skip raw binarized files
            .OrderBy(x => x)
            .ToList();

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
                Console.WriteLine($"HTM Training Cycle {cycle + 1}/{trainingCycles} completed.");
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

            double htmSim = ComputeHybridSimilarity(origSdr, htmReconSdr) * 100;
            double knnSim = ComputeHybridSimilarity(origSdr, knnReconSdr) * 100;

            Console.WriteLine($"Similarity Results for \"{name}\":");
            Console.WriteLine($" HTM Similarity: {htmSim:0.00}%");
            Console.WriteLine($" KNN Similarity: {knnSim:0.00}%\n");
        }
    }

    // Corrected SDR loading logic to handle multi-line files (0s and 1s grid)
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
