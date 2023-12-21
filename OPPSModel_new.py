
from OPPSModel import generate_initial_chromosome, sequencingConstraints, wn
import operator
import numpy as np
import random
from deap import base, creator, tools, gp, algorithms
from PIL import Image

# Define the primitive set
pset = gp.PrimitiveSet("MAIN", arity=0)
pset.addPrimitive(generate_initial_chromosome, arity=0)
pset.addPrimitive(operator.add, arity=2)
pset.addPrimitive(operator.mul, arity=2)
pset.addTerminal(0, bool)  # Terminal for False
pset.addTerminal(1, bool)  # Terminal for True

# Define the individual and population
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Define the genetic operators
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.register("select", tools.selTournament, tournsize=3)

# Evaluate the fitness of an individual
def eval_fitness(individual):
    x, c = gp.compile(individual, generate_initial_chromosome)
    
    # Check constraints
    if np.sum(np.matmul(x, wn)) <= 10 and np.sum(np.matmul(x, wn)) >= 6 and sequencingConstraints(x):
        return np.sum(c * x),
    else:
        return -float('inf'),  # Penalize individuals that violate constraints

toolbox.register("evaluate", eval_fitness)

# Main evolutionary loop
def main():
    population_size = 100
    generations = 50

    population = toolbox.population(n=population_size)
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.7, mutpb=0.3, ngen=generations, stats=None)

    # Extract the best individual from the final population
    best_ind = tools.selBest(offspring, 1)[0]

    # Print the best individual's tree
    print("Best Individual Tree:\n", best_ind)

if __name__ == "__main__":
    main()
