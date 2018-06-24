using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace GeneticAlgorithmChar
{
    class MainClass
    {
        private const bool DisplayOutputInConsole = true;

        private const string GoalString = "cameron";

        private static int Population = 8;
        private static float populationKeepRate = 0.5f;
        private static float MutateRate = 0.2f;

        private static int Epochs = 500;

        private static long HighestFitness = 0;
        private static long RunningAverageFitness = 0;
        private static List<long> allAverages = new List<long>();

        private static Random rand = new Random(System.Guid.NewGuid().GetHashCode());

        public static void Main(string[] args)
        {
            //variables
            var samples = GeneratePopulation();

            while (Epochs-- > 0)
            {
                //evaluate samples
                EvaluateSamples(samples);
                DisplaySamples("Generated population", samples);

                var generationAverage = CalculateGenerationAverage(samples);
                //to track average fitness per generation
                //allAverages.Add(generationAverage);

                RunningAverageFitness = CalculateTotalAverage(generationAverage, RunningAverageFitness);
                //to track average fitness per generation
                allAverages.Add(RunningAverageFitness);

                //sort the samples (negative for descending order)
                samples.Sort((sample1, sample2) => -sample1.Eval.CompareTo(sample2.Eval));

                DisplaySamples("Sorted Population:", samples);

                if (Epochs != 0) 
                {
                    //crossover
                    Crossover(ref samples);
                    DisplaySamples("After Crossover", samples);

                    //mutate
                    Mutate(ref samples);
                    DisplaySamples("After Mutation:", samples);
                }

                if (DisplayOutputInConsole)
                {
                    Console.WriteLine("Average fitness: {0}", RunningAverageFitness);
                    Console.WriteLine("Highest value seen: {0}\n", HighestFitness);
                }
            }

            Console.WriteLine("Evolutions completed");

            WriteAveragesToFile();

            Console.ReadLine();
        }

        static void DisplaySamples(string header, List<Sample> samplesIn)
        {
            if (!DisplayOutputInConsole)
                return;

            Console.WriteLine(header);
            for (int i = 0; i < samplesIn.Count; i++)
            {
                var sampleCodeString = string.Empty;
                for (int j = 0; j < samplesIn[i].Code.Length; j++)
                {
                    sampleCodeString += (char)('a' + samplesIn[i].Code[j]);
                }

                Console.WriteLine("Sample {0}: {1} => {2}", i, sampleCodeString, samplesIn[i].Eval);
            }
            Console.WriteLine();
        }

        static void WriteAveragesToFile()
        {
            using (StreamWriter writer = new StreamWriter("averages.txt"))
            {
                //print out generation averages in order
                for (int i = 0; i < allAverages.Count; i++)
                {
                    //Console.WriteLine("Generation {0}: {1}",i, allAverages[i]);
                    writer.WriteLine(allAverages[i]);
                }
            }
        }

        private static List<Sample> GeneratePopulation()
        {
            var samples = new List<Sample>();

            //generate sample population
            for (int i = 0; i < Population; i++)
            {
                var sample = new Sample(GoalString.Length);
                for (int j = 0; j < GoalString.Length; j++)
                {
                    //26 letters in the alphabet
                    sample.Code[j] = rand.Next() % 26;
                }

                samples.Add(sample);
            }

            return samples;
        }

        #region Calculations

        private static long CalculateGenerationAverage(List<Sample> samples)
        {
            //calculate this generations average fitness
            long tempAverage = 0;
            for (int i = 0; i < Population; i++)
            {
                tempAverage += samples[i].Eval;
            }
            tempAverage /= Population;
            return tempAverage;
        }

        private static long CalculateTotalAverage(long generationAverage, long totalAverage)
        {
            if (totalAverage > 0)
            {
                var tem = generationAverage + totalAverage;
                tem /= 2;
                totalAverage = tem;

            }
            else
            {
                totalAverage = generationAverage;
            }
            return totalAverage;
        }

        private static void EvaluateSamples(List<Sample> samples)
        {
            foreach (var samp in samples)
            {
                samp.Eval = EvaluateSample(samp.Code);
            }
        }

        static long EvaluateSample(int[] valueIn)
        {
            

            var numCorrect = 0;
            for (int i = 0; i < GoalString.Length; i++)
            {
                var correct = valueIn[i] == GoalString[i] - 'a';

                if (correct)
                    numCorrect++;
            }

            var fitness = Convert.ToInt64(Math.Pow(2, numCorrect));

            if (fitness > HighestFitness)
                HighestFitness = fitness;
            return fitness;
        }

        #endregion

        #region Genetics

        static void Crossover(ref List<Sample> samplesIn)
        {
            List<Sample> results = new List<Sample>();

            for (int i = 0; i < Population * populationKeepRate; i++)
            {
                //choose samples from the top 'keep' candidates
                Sample one = samplesIn[(int)((rand.Next() % Population) * populationKeepRate)];
                Sample two = samplesIn[(int)((rand.Next() % Population) * populationKeepRate)];

                Sample result1 = new Sample(GoalString.Length);
                Sample result2 = new Sample(GoalString.Length);

                //choose a cut point
                int cut = rand.Next() % GoalString.Length;

                Array.Copy(one.Code, 0, result1.Code, 0, cut);
                Array.Copy(two.Code, cut, result1.Code, cut, two.Code.Length-cut);

                Array.Copy(two.Code, 0, result2.Code, 0, cut);
                Array.Copy(one.Code, cut, result2.Code, cut, one.Code.Length - cut);

                results.Add(result1);
                results.Add(result2);
            }
            //store results in original vector
            for (int i = 0; i < results.Count; i++)
            {
                samplesIn[i] = results[i];
            }
        }

        static void Mutate(ref List<Sample> samplesIn)
        {
            for (int i = 0; i < samplesIn.Count - 1; i++)
            {
                //should we mutate
                if (((rand.Next() % 10) / 10f) < MutateRate)
                {
                    //which element to mutate
                    int mutateIndex = (rand.Next() % GoalString.Length);
                    samplesIn[i].Code[mutateIndex] = rand.Next() % 26;
                }
            }
        }

        #endregion
    }
}
