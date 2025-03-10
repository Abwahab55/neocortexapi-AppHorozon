using NeoCortexApi.Classifiers;
using NeoCortexApiSample;
using System;

class Program
{
    static void Main(string[] args)
    {
        Console.WriteLine("Starting HTM Image Recognition System...");

        // Step 1: Process images and generate SDRs
        ImageBinarizerSpatialPattern imageProcessor = new ImageBinarizerSpatialPattern();
        imageProcessor.Run();

        // Step 2: Initialize and train the KNN Classifier
        Console.WriteLine("Initializing KNN Classifier...");
        KnnImageClassifier knnClassifier = new KnnImageClassifier();
        knnClassifier.TrainClassifier("SDR_Values");

        // Step 3: Run the reconstruction
        Console.WriteLine("Initializing Image Reconstructor...");

        // Provide the correct paths for SDR values, Reconstructed Images, and Original Images
        string sdrFolder = @"C:\Software Engineering Project\neocortexapi\source\Samples\NeoCortexApiSample\bin\Debug\net8.0\SDR_Values"; // Path to SDR values
        string outputFolder = @"C:\Software Engineering Project\neocortexapi\source\Samples\NeoCortexApiSample\bin\Debug\net8.0\ReconstructedImages"; // Path to save reconstructed images
        string originalImagesFolder = @"C:\Software Engineering Project\neocortexapi\source\Samples\NeoCortexApiSample\bin\Debug\net8.0\Sample"; // Path to original images

        // Create an instance of KnnImageReconstructor
        //KnnImageReconstructor reconstructor = new KnnImageReconstructor(knnClassifier);

        // Run the reconstruction process with all required arguments
        //reconstructor.RunReconstruction(sdrFolder, outputFolder, originalImagesFolder);

        Console.WriteLine("Experiment completed.");
    }
}
