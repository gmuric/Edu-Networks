import numpy as np
from f_team import *
import array
import random
import numpy
from deap import algorithms
from deap import base
from deap import creator
from deap import tools


def genetic(soc_network,n,wplus,wminus,wneutr,scores,in_team):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", array.array, typecode='b', fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("indices", random.sample, range(n), n)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", f_total, soc_network=soc_network, n=n, wplus=wplus, wminus=wminus, wneutr=wneutr, in_team=in_team, scores=scores)
    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)


    random.seed(114)

    pop = toolbox.population(n=10)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=40, stats=stats, halloffame=hof, verbose=False)

    #calculating relative fitness: the ratio between the fitness of the result and the fitness of the random solution
    f_all=0
    for i in range(10):
        f = f_total(random.sample(range(n),n),soc_network,n,in_team, scores,wplus,wminus,wneutr)
        f_all += f[0]
        f_avg = f_all/10
    f_best = f_total(hof[0],soc_network,n,in_team, scores,wplus,wminus,wneutr)
    score = f_best/f_avg

    return (hof[0],score[0])

text_file = open("soc_net.dat", "r")
lines = text_file.read()
soc_network = np.matrix(lines)
n=21
wplus = 1
wminus = 0
wneutr = 0.5

scores = [5,5,5,1,1,1,2,3,3,5,5,5,5,1,1,1,4,4,4,1,1]
#x = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
in_team = 3

p = genetic(soc_network,n,wplus,wminus,wneutr,scores,in_team)
print(p)
#if __name__ == "__main__":
#    main()
#p = main()
#print(p[0][0])
#m = f_total(x,soc_network,n,in_team,scores,wplus,wminus,wneutr)
#print(m)
