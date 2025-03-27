using System;
using System.Collections.Generic;
using System.Linq;
using NeoCortexApi.Classifiers;
using NeoCortexApi.Entities;
using NeoCortexApi.Utility;

namespace NeoCortexApiSample
{
    public class KnnImageClassifier : IClassifier<int[], string>
    {
        private readonly List<int[]> trainingSdrs = new();
        private readonly int k;

        public KnnImageClassifier(int k = 5) => this.k = k;

        public void Learn(int[] input, Cell[] _) => trainingSdrs.Add(input);

        public int[] GetPredictedInputValue(Cell[] cells) =>
            GetPredictedInputValues(cells.Select(c => c.Index).ToArray(), 1).FirstOrDefault()?.PredictedInput;

        public List<ClassifierResult<int[]>> GetPredictedInputValues(int[] input, short howMany)
        {
            var results = trainingSdrs.Select(sdr => new ClassifierResult<int[]>
            {
                PredictedInput = sdr,
                Similarity = sdr.Zip(input, (a, b) => a == b ? 1 : 0).Sum()
            }).OrderByDescending(x => x.Similarity).Take(howMany).ToList();

            foreach (var res in results)
            {
                Console.WriteLine("\n[k-NN Prediction Similarity Metrics]");
                PrintSimilarityMetrics(input, res.PredictedInput);
            }

            return results;
        }

        private void PrintSimilarityMetrics(int[] original, int[] prediction)
        {
            double jaccard = MathHelpers.JaccardSimilarityofBinaryArrays(original, prediction);
            double cosine = ComputeCosineSimilarity(original, prediction);
            double hamming = ComputeHammingSimilarity(original, prediction);
            double hybrid = (jaccard + cosine + hamming) / 3.0;

            Console.WriteLine($"  Cosine:  {cosine:F4}");
            Console.WriteLine($"  Jaccard: {jaccard:F4}");
            Console.WriteLine($"  Hamming: {hamming:F4}");
            Console.WriteLine($"  Hybrid:  {hybrid:F4}");
        }

        private double ComputeCosineSimilarity(int[] a, int[] b)
        {
            double dot = a.Zip(b, (x, y) => x * y).Sum();
            double magA = Math.Sqrt(a.Sum(x => x * x));
            double magB = Math.Sqrt(b.Sum(y => y * y));
            return (magA == 0 || magB == 0) ? 0.0 : dot / (magA * magB);
        }

        private double ComputeHammingSimilarity(int[] a, int[] b)
        {
            return a.Zip(b, (x, y) => x == y ? 1 : 0).Sum() / (double)a.Length;
        }
    }
}
