using System;
using System.Collections.Generic;
using System.Linq;
using NeoCortexApi.Classifiers;
using NeoCortexApi.Entities;

namespace NeoCortexApiSample
{
    // This class implements a k-NN-based image classifier using the NeoCortexApi framework.
    // It maps between input SDRs (as int arrays) and their associated labels for classification and reconstruction.
    public class KnnImageClassifier : IClassifier<int[], string>
    {
        // Internal k-NN classifier instance from NeoCortexApi.
        private readonly KNeighborsClassifier<string, int[]> knn;

        // Constructor initializes the k-NN classifier.
        public KnnImageClassifier()
        {
            // The classifier doesn't require parameters like k or distance function;
            // it uses default settings and builds dynamically as data is learned.
            knn = new KNeighborsClassifier<string, int[]>();
        }

        // Learns a new input SDR and associates it with a label.
        // The label here is a stringified version of the SDR itself, used as a unique identifier.
        public void Learn(int[] input, Cell[] output)
        {
            string label = string.Join(",", input); // Flatten the SDR into a comma-separated string.
            knn.Learn(label, ConvertToCells(input)); // Teach the classifier using the label and corresponding cells.
        }

        // Predicts the most likely original SDR given a set of predictive cells (typically from HTM).
        // This is used during reconstruction to find the closest match to the predicted pattern.
        public int[] GetPredictedInputValue(Cell[] predictiveCells)
        {
            // Ask the k-NN model for the top-1 closest match.
            var results = knn.GetPredictedInputValues(predictiveCells, 1);

            // Extract the predicted label and convert it back into an int[] SDR.
            string bestLabel = results.FirstOrDefault()?.PredictedInput;
            return bestLabel?.Split(',').Select(int.Parse).ToArray() ?? Array.Empty<int>();
        }

        // Predicts multiple closest SDRs to the input, along with their similarity scores.
        // Useful for reconstruction when you want to explore more than one likely candidate.
        public List<ClassifierResult<int[]>> GetPredictedInputValues(int[] input, short howMany = 1)
        {
            // Convert the input SDR to a list of cells.
            var results = knn.GetPredictedInputValues(ConvertToCells(input), howMany);

            // Convert the predicted labels back to int[] and wrap in ClassifierResult for clarity.
            return results.Select(r => new ClassifierResult<int[]>
            {
                PredictedInput = r.PredictedInput.Split(',').Select(int.Parse).ToArray(),
                Similarity = r.Similarity
            }).ToList();
        }

        // Utility method to convert SDR indices into Cell objects used by the NeoCortexApi.
        private Cell[] ConvertToCells(int[] indices)
        {
            return indices.Select(i => new Cell { Index = i }).ToArray();
        }
    }
}
