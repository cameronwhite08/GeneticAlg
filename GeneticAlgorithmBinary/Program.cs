using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace GeneticAlgorithmBinary
{
    class MainClass
    {
        private const bool DisplayOutputInConsole = true;

        private static int Population = 8;
        private static int BitStringSize = 20;
        private static float populationKeepRate = 0.5f;
        private static float MutateRate = 0.2f;

        private static int Epochs = 100;

        private static long HighestFitness = 0;
        private static long RunningAverageFitness = 0;
        private static List<long> allAverages = new List<long>();

        private static Random rand = new Random(System.Guid.NewGuid().GetHashCode());

        public static void Main(string[] args)
        {
            //variables
            var samples = GenerateSamplePopulation();

            while (Epochs-- > 0)
            {
                //convert samples to decimal
                ConvertBinaryStringsToInts(samples);

                //evaluate samples to decimal
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
                Console.WriteLine("Sample {0}: {1} = {2} => {3}", i, samplesIn[i].Code, samplesIn[i].Value, samplesIn[i].Eval);
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

        private static List<Sample> GenerateSamplePopulation()
        {
            var samples = new List<Sample>();

            //generate sample population
            for (int i = 0; i < Population; i++)
            {
                var sample = new Sample();
                for (int j = 0; j < BitStringSize; j++)
                {
                    var chance = rand.Next() % 100;
                    if (chance >= 50)
                    {
                        sample.Code += "1";
                    }
                    else
                    {
                        sample.Code += "0";
                    }
                }

                samples.Add(sample);
            }

            return samples;
        }

        #region Calculations

        static void ConvertBinaryStringsToInts(List<Sample> samples)
        {
            foreach (var samp in samples)
            {
                samp.Value = BinaryStringToInt(samp.Code);
            }
        }

        //converting binary string to int equivilent
        static long BinaryStringToInt(string sampleIn)
        {
            var valu = 0L;
            var power = 0;

            //starting at end of string and working backwards
            for (int i = sampleIn.Length - 1; i >= 0; i--)
            {
                //get the correct base 2 value for this index in the binary string
                var powEval = Math.Pow(2, power++);
                //convert string value to int value
                var num = sampleIn[i] - '0';
                valu += Convert.ToInt64(powEval * num);
            }

            return valu;
        }

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
                samp.Eval = Evaluate(samp.Value);
            }
        }

        static long Evaluate(long valueIn)
        {
            var t = Convert.ToInt64(Math.Pow(valueIn, 2));

            if (t > HighestFitness)
                HighestFitness = t;
            return t;
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

                Sample result1 = new Sample();
                Sample result2 = new Sample();

                //choose a cut point
                int cut = rand.Next() % BitStringSize;

                //build new candidates
                result1.Code = one.Code.Substring(0, cut);
                result1.Code += two.Code.Substring(cut);

                result2.Code = two.Code.Substring(0, cut);
                result2.Code += one.Code.Substring(cut);

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
                    int mutateIndex = (rand.Next() % BitStringSize);
                    var aStringBuilder = new StringBuilder(samplesIn[i].Code);
                    if (samplesIn[i].Code[mutateIndex].Equals('0'))
                    {
                        aStringBuilder[mutateIndex] = '1';
                        samplesIn[i].Code = aStringBuilder.ToString();
                    }
                    else
                    {
                        aStringBuilder[mutateIndex] = '0';
                        samplesIn[i].Code = aStringBuilder.ToString();
                    }
                }
            }
        }

        #endregion
    }
}
