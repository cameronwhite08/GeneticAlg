using System;
using System.Collections.Generic;
using System.Linq;
using GeneticSharp.Domain;
using GeneticSharp.Domain.Chromosomes;
using GeneticSharp.Domain.Crossovers;
using GeneticSharp.Domain.Fitnesses;
using GeneticSharp.Domain.Mutations;
using GeneticSharp.Domain.Populations;
using GeneticSharp.Domain.Selections;
using GeneticSharp.Domain.Terminations;

namespace GeneticAlgWithLibrary
{
    class Program
    {
        private const string GoalString = "cameronwhite";

        private static long HighestFitness = 0;
        private static long RunningAverageFitness = 0;
        private static List<long> allAverages = new List<long>();

        public static void Main(string[] args)
        {
            //represents a sample in the population
            var chromosome = new FloatingPointChromosome(
                //min values array
                Enumerable.Repeat(0.0, GoalString.Length).ToArray(),
                //max values array
                Enumerable.Repeat(25.0, GoalString.Length).ToArray(),
                //bits nneded array
                Enumerable.Repeat(5, GoalString.Length).ToArray(),
                //decimals required array
                Enumerable.Repeat(0, GoalString.Length).ToArray()
                );

            //create the population
            var population = new Population(4, 8, chromosome);

            //define a fitness function
            var fitness = new FuncFitness((c) =>
            {
                var fc = c as FloatingPointChromosome;

                var values = fc.ToFloatingPoints();

                var numCorrect = 0;
                for (int i = 0; i < GoalString.Length; i++)
                {
                    var intVersion = Convert.ToInt32(values[i]);
                    if (intVersion == GoalString[i] - 'a')
                        numCorrect++;
                }

                return Convert.ToInt64(Math.Pow(2, numCorrect));
            });

            //select the top performers for reproduction
            var selection = new EliteSelection();

            //like is sounds, cut and splice chromosomes
            var crossover = new CutAndSpliceCrossover();
            
            //our numbers are internally represented as binary strings, randomly flip those bits
            var mutation = new FlipBitMutation();

            //stop mutating when there are 10 generations of stagnation
            var termination = new FitnessStagnationTermination(10);

            //put the genetic algorithm together
            var ga = new GeneticAlgorithm(
                population,
                fitness,
                selection,
                crossover,
                mutation);

            ga.Termination = termination;

            //print out the top performer of the population
            ga.GenerationRan += (sender, e) =>
            {
                var bestChromosome = ga.BestChromosome as FloatingPointChromosome;
                var bestFitness = Convert.ToInt32(bestChromosome.Fitness.Value);

                if (bestFitness != HighestFitness)
                {
                    HighestFitness = bestFitness;
                    var phenotype = bestChromosome.ToFloatingPoints();

                    var ints = phenotype.Select(Convert.ToInt32).ToArray();

                    var bestString = ConvertIntArrayToString(ints);

                    Console.WriteLine($"Best string: {bestString}");
                }
            };

            ga.TerminationReached += (sender, eventArgs) => { Console.WriteLine("Finished Evolving"); };

            ga.Start();

            Console.ReadKey();
        }

        private static string ConvertIntArrayToString(int[] code)
        {
            var sampleCodeString = string.Empty;
            for (int j = 0; j < code.Length; j++)
            {
                //ascii
                sampleCodeString += (char)('a' + code[j]);
            }

            return sampleCodeString;
        }
    }
}
