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
	    private static long averageFitness = 0;
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
					if (chance>=50)
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
					samples[i].Value = toInt(samples[i].Code);
				}

				//evaluate samples to decimal
				for (int i = 0; i < samples.Count; i++)
				{
					samples[i].Eval = evaluate(samples[i].Value);
				}

                Console.WriteLine("Generated population");
				ShowSamples(samples);

                //calculate this generations average fitness
                long tempAverage = 0;
                for (int i = 0; i < candidates; i++)
                {
                    tempAverage += samples[i].Eval;
                }
                tempAverage /= candidates;

                CalculateAverage(tempAverage);

                //sort according to fitness
                for (int i = 0; i < samples.Count; i++)
				{
					for (int j = i; j < samples.Count; j++)
					{
						if (samples[i].Eval < samples[j].Eval)
						{
							//switch
							Sample temp = samples[i];
							samples[i] = samples[j];
							samples[j] = temp;
						}
					}
				}

                Console.WriteLine("Sorted Population:");
				ShowSamples(samples);

				//crossover
				crossover(ref samples);

                Console.WriteLine("After Crossover");
				ShowSamples(samples);

				//mutate
				mutate(ref samples);

                Console.WriteLine("After Mutation:");
				ShowSamples(samples);

                Console.WriteLine("Average fitness: {0}", averageFitness);
			    Console.WriteLine("Highest value seen: {0}", highestValue);
			}

            //print out generation averages in order
		    for (int i = 0; i < allAverages.Count; i++)
		    {
		        //Console.WriteLine("Generation {0}: {1}",i, allAverages[i]);
		        Console.WriteLine(allAverages[i]);
		    }

		    Console.ReadLine();
		}

	    private static void CalculateAverage(long generationAverage)
	    {
            if(averageFitness > 0)
            {
                var tem = generationAverage + averageFitness;
                tem /= 2;
                averageFitness = tem;

            }
            else
            {
                averageFitness = generationAverage;
            }
            allAverages.Add(averageFitness);
	    }

	    static void ShowSamples(List<Sample> samplesIn)
		{
			for (int i = 0; i < samplesIn.Count; i++)
			{
				Console.WriteLine("Sample {0}: {1} = {2} => {3}", i, samplesIn[i].Code, samplesIn[i].Value, samplesIn[i].Eval);
			}
			Console.WriteLine();
		}

        //converting binary string to int equivilent
		static int toInt(string sampleIn)
		{
			var valu = 0;
			var power = 0;

            //starting at end of string and working backwards
			for (int i = sampleIn.Length-1; i >= 0; i--)
			{
                //get the correct base 2 value for this index in the binary string
                var powEval = Math.Pow(2, power++);
                //convert string value to int value
                var num = sampleIn[i] - '0';
                valu += Convert.ToInt32(powEval * num);
			}

			return valu;
		}


		static int evaluate(int valueIn)
		{
            var t = Convert.ToInt32(Math.Pow(valueIn, 2));
		    if (t > highestValue)
		        highestValue = t;
		    return t;
		}

		static void crossover(ref List<Sample> samplesIn)
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
				result1.Code = string.Empty;
				result2.Code = string.Empty;

				for (int j = 0; j < candidateSize; j++)
				{
					if (j<cut)
					{
						result1.Code += one.Code[j];
						result2.Code += two.Code[j];
					}
					else
					{
						result1.Code += two.Code[j];
						result2.Code += one.Code[j];
					}
				}
				results.Add(result1);
				results.Add(result2);
			}
			//store results in original vector
			for (int i = 0; i < results.Count; i++)
			{
				samplesIn[i] = results[i];
			}
		}

		static void mutate(ref List<Sample> samplesIn)
		{
			for (int i = 0; i < samplesIn.Count-1; i++)
			{
				//should we mutate
				if (((rand.Next()%10) / 10) < mutateRate)
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
	}
}
