using System.Collections.Generic;
using System.Linq;
using NeoCortexApi.Classifiers;
using NeoCortexApi.Entities;

namespace NeoCortexApiSample
{
    public class KnnImageClassifier : IClassifier<int[], string>
    {
        private readonly List<int[]> trainingSdrs = new();
        private readonly int k;

        public KnnImageClassifier(int k = 5)
        {
            this.k = k;
        }

        public void Learn(int[] input, Cell[] _)
        {
            trainingSdrs.Add(input);
        }

        public int[] GetPredictedInputValue(Cell[] cells)
        {
            return GetPredictedInputValues(cells.Select(c => c.Index).ToArray(), 1)
                .FirstOrDefault()?.PredictedInput;
        }

        public List<ClassifierResult<int[]>> GetPredictedInputValues(int[] input, short howMany)
        {
            return trainingSdrs.Select(sdr => new ClassifierResult<int[]>
            {
                PredictedInput = sdr,
                Similarity = sdr.Zip(input, (a, b) => a == b ? 1 : 0).Sum()
            })
            .OrderByDescending(x => x.Similarity)
            .Take(howMany)
            .ToList();
        }
    }
}
