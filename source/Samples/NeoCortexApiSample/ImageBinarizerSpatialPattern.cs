using System;
using System.IO;
using System.Linq;
using System.Drawing;
using Daenet.ImageBinarizerLib;
using Daenet.ImageBinarizerLib.Entities;
using NeoCortexApi;
using NeoCortexApi.Entities;

namespace NeoCortexApiSample
{
    /// <summary>
    /// This class loads grayscale images, binarizes them into black-and-white (0 or 1),
    /// and feeds them into an HTM Spatial Pooler to extract sparse distributed representations (SDRs).
    /// It saves these SDRs as text files for later use (e.g., classification or reconstruction).
    /// </summary>
    public class ImageBinarizerSpatialPattern
    {
        private readonly string trainingFolder;
        private readonly SpatialPooler spatialPooler;
        private readonly Connections connections;
        private const int ImageSize = 64; // Width and height of the images in pixels (assumed square)

        /// <summary>
        /// Constructor initializes the spatial pooler and its connection config.
        /// </summary>
        /// <param name="trainingFolder">Path to folder containing training images.</param>
        public ImageBinarizerSpatialPattern(string trainingFolder)
        {
            if (!Directory.Exists(trainingFolder))
                throw new ArgumentException("Invalid training folder path.");

            this.trainingFolder = trainingFolder;

            // Set up HTM configuration for Spatial Pooler
            connections = new Connections(new HtmConfig
            {
                ColumnDimensions = new[] { ImageSize, ImageSize }, // Spatial pooler grid size
                InputDimensions = new[] { ImageSize, ImageSize },  // Input image dimensions
                NumInputs = ImageSize * ImageSize,                 // Total number of input bits
                PotentialPct = 0.85,                               // Fraction of input bits each column can potentially be connected to
                NumActiveColumnsPerInhArea = 100,                  // Number of active columns to maintain sparsity
                SynPermInactiveDec = 0.005,                        // How much to decrease permanence for inactive synapses
                SynPermActiveInc = 0.04,                           // How much to increase permanence for active synapses
                SynPermConnected = 0.2                             // Permanence threshold to consider synapse as connected
            });

            // Initialize the Spatial Pooler with configured connections
            spatialPooler = new SpatialPooler();
            spatialPooler.Init(connections);
        }

        /// <summary>
        /// Runs the binarization and SDR generation pipeline for all images in the training folder.
        /// </summary>
        public void Run()
        {
            var images = Directory.GetFiles(trainingFolder, "*.png"); // Get all PNG images in folder
            var sdrOutputFolder = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "SDR_Values");
            Directory.CreateDirectory(sdrOutputFolder); // Ensure output folder exists

            foreach (var imagePath in images)
            {
                var imageName = Path.GetFileNameWithoutExtension(imagePath);

                // Step 1: Binarize the image (convert to 0/1 pixel values)
                var binarizedPixels = BinarizeImage(imagePath, sdrOutputFolder, imageName);
                if (binarizedPixels == null) continue;

                // Step 2: Feed binarized image into the Spatial Pooler
                var activeColumns = new int[connections.HtmConfig.NumColumns];
                spatialPooler.compute(binarizedPixels, activeColumns, true); // Learn = true

                // Step 3: Save the SDR (active columns) to a text file
                File.WriteAllText(Path.Combine(sdrOutputFolder, $"{imageName}.txt"),
                    string.Join(",", activeColumns.Select(bit => bit.ToString())));
            }
        }

        /// <summary>
        /// Binarizes a grayscale image by thresholding it, saves the result, and returns a flattened binary array.
        /// </summary>
        /// <param name="path">Path to the input image file.</param>
        /// <param name="folder">Folder to save the binarized image result.</param>
        /// <param name="name">Base name to use for output file.</param>
        /// <returns>Flattened array of 0s and 1s representing binarized image.</returns>
        private int[] BinarizeImage(string path, string folder, string name)
        {
            var binParams = new BinarizerParams
            {
                InputImagePath = path,
                OutputImagePath = Path.Combine(folder, $"{name}_binarized.txt"),
                GreyScale = true,
                ImageWidth = ImageSize,
                ImageHeight = ImageSize,
                GreyThreshold = 128 // Pixels above this value become white (1), below become black (0)
            };

            // Run binarization using the Daenet library
            new ImageBinarizer(binParams).Run();

            // Read the binarized result and flatten it into a 1D binary array
            var lines = File.ReadAllLines(binParams.OutputImagePath);
            return lines.SelectMany(line => line.Select(ch => ch == '1' ? 1 : 0)).ToArray();
        }
    }
}
