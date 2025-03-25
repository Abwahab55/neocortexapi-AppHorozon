using Daenet.ImageBinarizerLib.Entities;
using Daenet.ImageBinarizerLib;
using System.IO;
using System.Linq;

namespace NeoCortexApiSample
{
    public class ImageBinarizationUtils
    {
        public static string BinarizeImages(int imageWidth, int imageHeight, string destinationPath, string imagePath)
        {
            string binaryImage = $"{destinationPath}.txt";

            if (File.Exists(binaryImage))
                File.Delete(binaryImage);

            ImageBinarizer imageBinarizer = new ImageBinarizer(new BinarizerParams
            {
                RedThreshold = 200,
                GreenThreshold = 200,
                BlueThreshold = 200,
                ImageWidth = imageWidth,
                ImageHeight = imageHeight,
                InputImagePath = imagePath,
                OutputImagePath = binaryImage
            });

            imageBinarizer.Run();

            var binaryData = File.ReadAllLines(binaryImage)
                                 .Select(line => new string(line.Select(ch => ch == '0' ? '1' : '0').ToArray()))
                                 .ToArray();

            File.WriteAllLines(binaryImage, binaryData);
            return binaryImage;
        }
    }
}
