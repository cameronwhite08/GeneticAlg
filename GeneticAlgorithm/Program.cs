using System;
using System.Collections.Generic;
using System.Text;

namespace GeneticAlgorithm
{
    //max value seen so far
    //avg fitness per generation
    class MainClass
    {
        static int candidates = 8;
        static int candidateSize = 12;
        static float keep = 0.5f;
        static float mutateRate = 0.2f;
        private static int loops = 100;
        private static long highestValue = 0;
        private static long totalAverageFitness = 0;
        private static List<long> allAverages = new List<long>();

        static Random rand = new Random();

        public static void Main(string[] args)
        {
            //variables
            var samples = new List<Sample>();

            //generate sample population
            for (int i = 0; i < candidates; i++)
            {
                var sample = new Sample();
                for (int j = 0; j < candidateSize; j++)
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

            while (loops-- > 0)
            {
                //convert samples to decimal
                for (int i = 0; i < samples.Count; i++)
                {
                    samples[i].Value = BinaryStringToInt(samples[i].Code);
                }

                //evaluate samples to decimal
                for (int i = 0; i < samples.Count; i++)
                {
                    samples[i].Eval = Evaluate(samples[i].Value);
                }

                DisplaySamples("Generated population", samples);

                var generationAverage = CalculateGenerationAverage(samples);
                //to track average fitness per generation
                //allAverages.Add(generationAverage);

                totalAverageFitness = CalculateTotalAverage(generationAverage, totalAverageFitness);
                //to track average fitness per generation
                allAverages.Add(totalAverageFitness);

                //sort the samples (negative for descending order)
                samples.Sort((sample1, sample2) => -sample1.Eval.CompareTo(sample2.Eval));

                DisplaySamples("Sorted Population:", samples);

                if (loops != 0) 
                {
                    //crossover
                    Crossover(ref samples);

                    DisplaySamples("After Crossover", samples);

                    //mutate
                    Mutate(ref samples);

                    DisplaySamples("After Mutation:", samples);
                }
                
                Console.WriteLine("Average fitness: {0}", totalAverageFitness);
                Console.WriteLine("Highest value seen: {0}\n", highestValue);
            }

            //print out generation averages in order
            for (int i = 0; i < allAverages.Count; i++)
            {
                //Console.WriteLine("Generation {0}: {1}",i, allAverages[i]);
                Console.WriteLine(allAverages[i]);
            }

            Console.ReadLine();
        }

        static void DisplaySamples(string header, List<Sample> samplesIn)
        {
            Console.WriteLine(header);
            for (int i = 0; i < samplesIn.Count; i++)
            {
                Console.WriteLine("Sample {0}: {1} = {2} => {3}\n", i, samplesIn[i].Code, samplesIn[i].Value, samplesIn[i].Eval);
            }
        }
        
        #region Calculations

        //converting binary string to int equivilent
        static int BinaryStringToInt(string sampleIn)
        {
            var valu = 0;
            var power = 0;

            //starting at end of string and working backwards
            for (int i = sampleIn.Length - 1; i >= 0; i--)
            {
                //get the correct base 2 value for this index in the binary string
                var powEval = Math.Pow(2, power++);
                //convert string value to int value
                var num = sampleIn[i] - '0';
                valu += Convert.ToInt32(powEval * num);
            }

            return valu;
        }

        private static long CalculateGenerationAverage(List<Sample> samples)
        {
            //calculate this generations average fitness
            long tempAverage = 0;
            for (int i = 0; i < candidates; i++)
            {
                tempAverage += samples[i].Eval;
            }
            tempAverage /= candidates;
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

        static int Evaluate(int valueIn)
        {
            var t = Convert.ToInt32(Math.Pow(valueIn, 2));
            if (t > highestValue)
                highestValue = t;
            return t;
        }

        #endregion

        #region Genetics

        static void Crossover(ref List<Sample> samplesIn)
        {
            List<Sample> results = new List<Sample>();

            for (int i = 0; i < candidates * keep; i++)
            {
                //choose samples from the top 'keep' candidates
                Sample one = samplesIn[(int)((rand.Next() % candidates) * keep)];
                Sample two = samplesIn[(int)((rand.Next() % candidates) * keep)];

                Sample result1 = new Sample();
                Sample result2 = new Sample();

                //choose a cut point
                int cut = rand.Next() % candidateSize;

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
                if (((rand.Next() % 10) / 10) < mutateRate)
                {
                    //which element to mutate
                    int mutateIndex = (rand.Next() % candidateSize);

                    if (samplesIn[i].Code[mutateIndex].Equals('0'))
                    {
                        var aStringBuilder = new StringBuilder(samplesIn[i].Code);
                        aStringBuilder.Remove(i, 1);
                        aStringBuilder.Insert(i, "1");
                        samplesIn[i].Code = aStringBuilder.ToString();
                    }
                    else
                    {
                        var aStringBuilder = new StringBuilder(samplesIn[i].Code);
                        aStringBuilder.Remove(i, 1);
                        aStringBuilder.Insert(i, "0");
                        samplesIn[i].Code = aStringBuilder.ToString();
                    }
                }
            }
        }

        #endregion
    }
}
