using System;
using System.Collections.Generic;
using System.Linq;
using NeoCortexApi.Classifiers;
using NeoCortexApi.Entities;

namespace NeoCortexApiSample
{
    public class KnnImageClassifier : IClassifier<int[], string>
    {
        private readonly KNeighborsClassifier<string, int[]> knn;

        public KnnImageClassifier()
        {
            knn = new KNeighborsClassifier<string, int[]>(); // no constructor args needed
        }

        // Learn SDR with associated label
        public void Learn(int[] input, Cell[] output)
        {
            string label = string.Join(",", input); // You may replace with filename if available
            knn.Learn(label, ConvertToCells(input));
        }

        // Predict the most likely SDR from cell indices
        public int[] GetPredictedInputValue(Cell[] predictiveCells)
        {
            var results = knn.GetPredictedInputValues(predictiveCells, 1);
            string bestLabel = results.FirstOrDefault()?.PredictedInput;
            return bestLabel?.Split(',').Select(int.Parse).ToArray() ?? Array.Empty<int>();
        }

        // Predict top-k closest SDRs (used in reconstruction)
        public List<ClassifierResult<int[]>> GetPredictedInputValues(int[] input, short howMany = 1)
        {
            var results = knn.GetPredictedInputValues(ConvertToCells(input), howMany);
            return results.Select(r => new ClassifierResult<int[]>
            {
                PredictedInput = r.PredictedInput.Split(',').Select(int.Parse).ToArray(),
                Similarity = r.Similarity
            }).ToList();
        }

        private Cell[] ConvertToCells(int[] indices)
        {
            return indices.Select(i => new Cell { Index = i }).ToArray();
        }
    }
}
