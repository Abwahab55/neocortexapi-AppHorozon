using System;
using System.Collections.Generic;
using System.Linq;
using NeoCortexApi.Classifiers;
using NeoCortexApi.Entities;

namespace NeoCortexApiSample
{
    public class KnnImageClassifier : IClassifier<int[], string>
    {
        private Dictionary<string, List<int[]>> trainingData = new Dictionary<string, List<int[]>>();
        private int _k = 10;

        public void Learn(int[] input, Cell[] output)
        {
            string label = string.Join(",", input);
            if (!trainingData.ContainsKey(label))
                trainingData[label] = new List<int[]>();

            if (!trainingData[label].Any(sdr => sdr.SequenceEqual(input)))
            {
                trainingData[label].Add(input);
            }
        }

        public int[] GetPredictedInputValue(Cell[] unclassifiedCells)
        {
            if (unclassifiedCells == null || unclassifiedCells.Length == 0)
            {
                return Array.Empty<int>();
            }

            int[] inputSdr = unclassifiedCells.Select(c => c.Index).ToArray();
            var predictedResults = GetPredictedInputValues(inputSdr, 1);
            return predictedResults.Count > 0 ? predictedResults[0].PredictedInput : Array.Empty<int>();
        }

        public List<ClassifierResult<int[]>> GetPredictedInputValues(int[] inputSdr, short howMany = 3)
        {
            var results = new List<ClassifierResult<int[]>>();
            foreach (var kv in trainingData)
            {
                var nearestNeighbors = kv.Value
                    .OrderByDescending(storedSdr => storedSdr.Zip(inputSdr, (a, b) => a == b ? 1 : 0).Sum())
                    .Take(Math.Min(_k, kv.Value.Count))
                    .ToList();

                if (nearestNeighbors.Count == 0)
                {
                    continue;
                }

                results.Add(new ClassifierResult<int[]>
                {
                    PredictedInput = nearestNeighbors[0],
                    Similarity = 0 // Placeholder, as similarity is computed in Program.cs
                });
            }

            return results.OrderByDescending(r => r.Similarity).Take(howMany).ToList();
        }
    }
}
