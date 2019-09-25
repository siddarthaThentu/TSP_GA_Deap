import random

from deap import base
from deap import creator
from deap import tools
import pandas as pd
import statistics

creator.create("Total_Population", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.Total_Population)

toolbox = base.Toolbox()
Individual_Size = 8

# permutation setup for individual
toolbox.register("indices", random.sample, range(Individual_Size), Individual_Size)
toolbox.register("individual", tools.initIterate, creator.Individual,toolbox.indices)
# population setup
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Assign numbers to column names in a dictionary

dic = {}

count=-1
dist=pd.read_csv("C:/Users/admin/Desktop/TS_Distances_Between_Cities.csv")

for col in dist.columns.values:
    dic[count]=col
    count+=1


distances = pd.read_csv("TS_Distances_Between_Cities.csv",header=None, skipfooter=1,skiprows=1,usecols=range(1,9))

# the goal ('fitness') function to be minimized
def EVALUATE(individual):
    dis=[]

    for i in range(0,len(individual)-1):
        new_dis=distances.iloc[individual[i],individual[i+1]]
        dis.append(new_dis)
    return sum(dis),

# Creating an alias for evalTSP by registering
toolbox.register("evaluate", EVALUATE)

# operator for selecting individuals for breeding the next
# generation: each individual of the current generation
# is replaced by the 'fittest' (best) of three individuals
# drawn randomly from the current generation.
toolbox.register("select", tools.selTournament, tournsize=3)

# register the crossover operator
toolbox.register("mate", tools.cxOrdered)

# register a mutation operator with a probability to
# shuffle randomly with probability of 0.5
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.5)


def main():
    random.seed(100)

    # create an initial population of 500 individuals (where
    # each individual is a list of integers)
    populaTion = toolbox.population(n=100)

    #Probability of Cross Over
    CrossProb = 0.3
    
    #Probabiliy of Mutation
    MutateProb = 0.2

    print("Start of evolution")

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, populaTion))
    for ind, fit in zip(populaTion, fitnesses):
        ind.fitness.values = fit

    # Extracting all the fitnesses of an individual
    fits = [ind.fitness.values[0] for ind in populaTion]

    # Variable keeping track of the number of generations
    g = 0
    file = open('C:/Users/admin/Desktop/Siddartha_Thentu_GA_TS_Info.txt','w')

    # Begin the evolution
    while min(fits) > 10000 and g < 100:
        # A new generation
        g = g + 1


        # Select the next generation individuals
        offspring = toolbox.select(populaTion, len(populaTion))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CrossProb
            if random.random() < CrossProb:
                toolbox.mate(child1, child2)

                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:

            # mutate an individual with probability MutateProb
            if random.random() < MutateProb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_Individual = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_Individual)
        for ind, fit in zip(invalid_Individual, fitnesses):
            ind.fitness.values = fit

        # Replace the population with the new offspring population
        populaTion[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in populaTion]


        file.write(str(g)+'. '+ 'Population Size : '+str(len(populaTion))+' for iteration '+str(g)+'\n')
        file.write('Average Fitness Score : '+str(statistics.mean(fits))+'\n' )
        file.write('Median Fitness Score : '+str(statistics.median(fits))+'\n' )
        file.write('STD of Fitness Score : '+str(statistics.stdev(fits))+'\n' )
        file.write('Size of selected subset of population : '+str(len(invalid_Individual))+'\n' )
        file.write('\n')


    print("End of evolution ")

    bestIndividual = tools.selBest(populaTion, 1)[0]
    file2 = open('C:/Users/admin/Desktop/Siddartha_Thentu_GA_TS_Result.txt','w')
    counter=1
    for ind in bestIndividual:

        file2.write(str(counter)+'. '+str(ind+1) + '/' + str(dic[ind])+'\n')
        counter+=1

if __name__ == "__main__":
    main()