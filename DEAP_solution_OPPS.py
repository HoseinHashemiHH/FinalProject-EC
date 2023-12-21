
from OPPSModel import generate_initial_chromosome
import operator
import random
import numpy as np
from deap import creator, base, tools, gp, algorithms

# Assume your existing code and functions are here, including the definition of generate_initial_chromosome

# Define DEAP Types and Fitness Function
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

# Define Genetic Operators
toolbox = base.Toolbox()

# Define terminals and functions
pset = gp.PrimitiveSetTyped("MAIN", [bool, bool], float)
pset.addPrimitive(np.add, [float, float], float)
pset.addPrimitive(np.subtract, [float, float], float)
pset.addPrimitive(np.multiply, [float, float], float)
pset.addPrimitive(np.divide, [float, float], float)
pset.addTerminal(0.0, float)
pset.addTerminal(1.0, float)

# Register the operators
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def evalObjective(individual):
    x, c = generate_initial_chromosome()
    objective_value = np.sum(c * x)  # Adjust as per your objective function
    return objective_value,

toolbox.register("evaluate", evalObjective)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    population = toolbox.population(n=4)
    generations = 4
    cxpb, mutpb = 0.7, 0.2

    toolbox.register("mate", tools.cxBlend, alpha=0.5)


    algorithms.eaMuPlusLambda(population, toolbox, mu=100, lambda_=200, cxpb=cxpb, mutpb=mutpb, ngen=generations,
                              stats=None, halloffame=None, verbose=True)

    best_individual = tools.selBest(population, 1)[0]
    best_solution = generate_initial_chromosome(best_individual)
    print("Best Solution:", best_solution)
    print("Best Fitness:", best_individual.fitness.values[0])

if __name__ == "__main__":
    main()