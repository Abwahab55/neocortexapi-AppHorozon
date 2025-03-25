using System;
using System.Collections.Generic;
using System.Linq;
using NeoCortexApi.Classifiers;
using NeoCortexApi.Entities;
using NeoCortexApi.Utility;

namespace NeoCortexApiSample
{
    /// <summary>
    /// An image reconstruction classifier using k-Nearest Neighbors.
    /// It stores all input SDRs and, for a given new SDR, finds the most similar stored patterns.
    /// Implements IClassifier with input type int[] (SDR) and output type string (not used for labeling directly).
    /// </summary>
    public class KnnImageClassifier : IClassifier<int[], string>
    {
        // Mapping from a "label" (here we use the SDR string representation as a key) to stored SDR patterns.
        private Dictionary<string, List<int[]>> trainingData = new Dictionary<string, List<int[]>>();
        private int _k = 10;  // Number of neighbors to consider for predictions

        /// <summary>
        /// Stores the given input SDR pattern in the training dataset (if not already present).
        /// Uses the SDR's string representation as a unique key.
        /// </summary>
        public void Learn(int[] input, Cell[] output)
        {
            string label = string.Join(",", input);
            if (!trainingData.ContainsKey(label))
                trainingData[label] = new List<int[]>();

            // Only add if this exact SDR has not been added before (avoid duplicates)
            if (!trainingData[label].Any(sdr => sdr.SequenceEqual(input)))
            {
                trainingData[label].Add(input);
                Console.WriteLine($"[LEARN] Stored SDR of length {input.Length}. Total entries for key {label}: {trainingData[label].Count}");
            }
            else
            {
                Console.WriteLine($"[WARNING] Duplicate SDR detected for key {label}, skipping.");
            }
        }

        /// <summary>
        /// Given an unclassified set of active cells (from SP output), returns the single best matching stored SDR (or empty array if none).
        /// </summary>
        public int[] GetPredictedInputValue(Cell[] unclassifiedCells)
        {
            Console.WriteLine("\n[DEBUG] Entered GetPredictedInputValue() for k-NN classifier");
            if (unclassifiedCells == null || unclassifiedCells.Length == 0)
            {
                Console.WriteLine("[WARNING] No unclassified cells provided to k-NN classifier!");
                return Array.Empty<int>();
            }

            // Convert Cell[] to input SDR indices
            int[] inputSdr = unclassifiedCells.Select(c => c.Index).ToArray();
            Console.WriteLine($"[INFO] Input SDR extracted (length {inputSdr.Length}, first 10 bits: {string.Join(",", inputSdr.Take(10))})");

            var predictedResults = GetPredictedInputValues(inputSdr, 1);
            if (predictedResults == null || predictedResults.Count == 0)
            {
                Console.WriteLine("[WARNING] k-NN classifier found no predictions.");
                return Array.Empty<int>();
            }

            Console.WriteLine("[SUCCESS] k-NN prediction generated.");
            Console.WriteLine($"[INFO] Predicted SDR (length {predictedResults[0].PredictedInput.Length}, first 10 bits: {string.Join(",", predictedResults[0].PredictedInput.Take(10))})");
            return predictedResults[0].PredictedInput;
        }

        /// <summary>
        /// Given an input SDR, finds up to 'howMany' most similar SDRs from the training data.
        /// Returns a list of ClassifierResult with predicted SDRs and similarity scores.
        /// </summary>
        public List<ClassifierResult<int[]>> GetPredictedInputValues(int[] inputSdr, short howMany = 3)
        {
            Console.WriteLine("\n[DEBUG] Entered GetPredictedInputValues() for k-NN classifier");
            Console.WriteLine($"[INFO] Finding nearest neighbors for SDR (first 10 bits: {string.Join(",", inputSdr.Take(10))})");

            var results = new List<ClassifierResult<int[]>>();

            // Iterate through each stored SDR list in training data (each key corresponds to one unique SDR pattern)
            foreach (var kv in trainingData)
            {
                if (kv.Value.Count == 0) continue;
                // For k-NN, if there are multiple identical SDRs stored under the same key, we can consider them all.
                // Here, compute similarity of the input to each SDR under this key and take the nearest neighbor(s).
                var nearestNeighbors = kv.Value
                    .Select(storedSdr => new
                    {
                        Sdr = storedSdr,
                        Similarity = ComputeHybridSimilarity(storedSdr, inputSdr)
                    })
                    .OrderByDescending(x => x.Similarity)
                    .Take(Math.Min(_k, kv.Value.Count))
                    .ToList();

                if (nearestNeighbors.Count == 0)
                    continue;

                // Use the first neighbor (highest similarity) as representative for this key.
                // We could also decide based on majority vote across neighbors if multiple, but we'll handle voting later in reconstruction.
                double avgSimilarity = nearestNeighbors.Average(x => x.Similarity);
                Console.WriteLine($"[SIMILARITY] Avg similarity for key {kv.Key.Substring(0, Math.Min(20, kv.Key.Length))}...: {avgSimilarity * 100:F2}%");

                results.Add(new ClassifierResult<int[]>
                {
                    PredictedInput = nearestNeighbors[0].Sdr,
                    Similarity = avgSimilarity
                });
            }

            if (results.Count == 0)
            {
                Console.WriteLine("[WARNING] No neighbors found in training data.");
                return new List<ClassifierResult<int[]>>();
            }

            // Return the top 'howMany' results with highest similarity
            var topResults = results.OrderByDescending(r => r.Similarity).Take(howMany).ToList();
            Console.WriteLine($"[SUCCESS] k-NN classifier returning {topResults.Count} top match(es).");
            return topResults;
        }

        /// <summary>
        /// Computes a hybrid similarity between two binary SDRs by combining Jaccard, Cosine, and Hamming measures.
        /// Returns a value between 0 and 1 (higher is more similar).
        /// </summary>
        private static double ComputeHybridSimilarity(int[] a, int[] b)
        {
            if (a.Length != b.Length)
            {
                // If lengths differ, we cannot compare properly.
                return 0;
            }
            double jaccard = MathHelpers.JaccardSimilarityofBinaryArrays(a, b);
            double cosineSim = 1.0 - CosineDistance(a, b);
            double hammingSim = 1.0 - (double)HammingDistance(a, b) / a.Length;
            return (jaccard + cosineSim + hammingSim) / 3.0;
        }

        /// <summary>
        /// Calculates the Hamming distance (number of different bits) between two equal-length SDRs.
        /// </summary>
        private static int HammingDistance(int[] a, int[] b) => a.Zip(b, (x, y) => x == y ? 0 : 1).Sum();

        /// <summary>
        /// Calculates the cosine distance (1 - cosine similarity) between two SDRs.
        /// </summary>
        private static double CosineDistance(int[] a, int[] b)
        {
            double dot = a.Zip(b, (x, y) => x * y).Sum();
            double magA = Math.Sqrt(a.Sum(x => x * x));
            double magB = Math.Sqrt(b.Sum(y => y * y));
            if (magA == 0 || magB == 0) return 1.0;
            return 1.0 - (dot / (magA * magB));
        }
    }
}
