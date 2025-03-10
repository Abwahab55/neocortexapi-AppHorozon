using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using NeoCortexApi.Entities;
using NeoCortexApi.Classifiers;

namespace NeoCortexApi.Classifiers
{
    public class KnnImageClassifier : IClassifier<int[], string>
    {
        private Dictionary<string, List<int[]>> sdrMap = new Dictionary<string, List<int[]>>();

        public void Learn(int[] input, Cell[] output)
        {
            string label = string.Join(",", input);
            if (!sdrMap.ContainsKey(label))
            {
                sdrMap[label] = new List<int[]>();
            }
            sdrMap[label].Add(output.Select(c => c.Index).ToArray());
        }

        public List<ClassifierResult<int[]>> GetPredictedInputValues(int[] predictiveCells, short howMany = 1)
        {
            var results = new List<ClassifierResult<int[]>>();
            foreach (var kvp in sdrMap)
            {
                var similarity = CalculateSimilarity(predictiveCells, kvp.Value.FirstOrDefault());
                results.Add(new ClassifierResult<int[]> { PredictedInput = kvp.Value.First(), Similarity = similarity });
            }
            return results.OrderByDescending(r => r.Similarity).Take(howMany).ToList();
        }

        public int[] GetPredictedInputValue(Cell[] predictiveCells)
        {
            var bestMatch = GetPredictedInputValues(predictiveCells.Select(c => c.Index).ToArray(), 1).FirstOrDefault();
            return bestMatch?.PredictedInput ?? new int[0];
        }

        public void TrainClassifier(string sdrFolder)
        {
            var sdrFiles = Directory.GetFiles(sdrFolder, "sdr_*.csv");

            if (sdrFiles.Length == 0)
            {
                Console.WriteLine("Error: No SDR files found for training.");
                return;
            }

            Console.WriteLine($"Training KNN Classifier with {sdrFiles.Length} SDR files...");

            foreach (var file in sdrFiles)
            {
                string imageName = Path.GetFileNameWithoutExtension(file).Replace("sdr_", "");
                int[] sdrValues = File.ReadAllLines(file)
                                      .SelectMany(line => line.Split(',')
                                      .Select(value => int.TryParse(value, out int num) ? num : -1))
                                      .Where(num => num >= 0)
                                      .ToArray();

                if (sdrValues.Length == 0)
                {
                    Console.WriteLine($"Warning: Empty SDR file detected for {imageName}, skipping...");
                    continue;
                }

                // Convert SDR values into Cell[]
                Cell[] cells = sdrValues.Select(index => new Cell() { Index = index }).ToArray();

                // Train KNN Classifier
                Learn(sdrValues, cells);

                Console.WriteLine($"KNN Learning: {imageName} -> SDR (first 50): {string.Join(",", sdrValues.Take(50))}...");
            }

            Console.WriteLine("KNN Classifier Training Completed.");
        }

        private double CalculateSimilarity(int[] testSDR, int[] knownSDR)
        {
            if (testSDR == null || knownSDR == null) return 0;
            return testSDR.Intersect(knownSDR).Count() / (double)testSDR.Length;
        }
    }
}
