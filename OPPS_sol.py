
from OPPSModel import generate_initial_chromosome
import numpy as np
from deap import creator, base, tools, gp, algorithms
import networkx as nx
import matplotlib.pyplot as plt
import pygraphviz as pgv
from PIL import Image

# Assume your existing code and functions are here, including the definition of generate_initial_chromosome

# Define DEAP Types and Fitness Function
creator.create("FitnessMin", base.Fitness, weights=(1.0,))
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
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

# Initialization of x and c outside the evaluation function
initial_x, initial_c = generate_initial_chromosome()

def evalObjective(individual):
    objective_value = np.sum(initial_c * initial_x)  # Adjust as per your objective function
    return objective_value,

toolbox.register("evaluate", evalObjective)
def cxBlend(ind1, ind2, alpha):
    size = min(len(ind1), len(ind2))
    for i in range(size):
        if isinstance(ind1[i], float) and isinstance(ind2[i], float):
            ind1[i] = (1. - alpha) * ind1[i] + alpha * ind2[i]
            ind2[i] = alpha * ind1[i] + (1. - alpha) * ind2[i]
        elif isinstance(ind1[i], np.ndarray) and isinstance(ind2[i], np.ndarray):
            # Handle numpy arrays if needed
            pass
    return ind1, ind2

toolbox.register("mate", cxBlend, alpha=0.5)
toolbox.register("expr_mut", gp.genFull, min_=1, max_=3)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    population = toolbox.population(n=100)
    generations = 400
    cxpb, mutpb = 0.7, 0.2

    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind)

    for gen in range(1, generations + 1):
        offspring = algorithms.varAnd(population, toolbox, cxpb, mutpb)
        fitness_values = list(map(toolbox.evaluate, offspring))
        for ind, fit in zip(offspring, fitness_values):
            ind.fitness.values = fit

        population[:] = tools.selBest(offspring + population, len(population))

        best_individual = tools.selBest(population, 1)[0]
        print(f"Generation {gen} - Best Fitness: {best_individual.fitness.values[0]}")
        # Print the tree of the best individual
     # Compile the best individual
    compiled_function = toolbox.compile(expr=best_individual)

    # Create a symbolic expression from the compiled function
    best_tree_str = str(best_individual)
    best_tree = gp.PrimitiveTree.from_string(best_tree_str, pset)

    # Create a graph from the symbolic expression
    graph = gp.graph(best_tree)

    # Print the graph
    print("\nBest Individual Tree:")
    print(graph)

    # -------------------------------Draw the graph-----------------------
    nodes, edges, labels = gp.graph(best_tree)

    graph = pgv.AGraph(directed=True)
    graph.add_nodes_from(nodes)

    # Add labels with function names
    for node, label in labels.items():
        graph.get_node(node).attr['label'] = label

    graph.add_edges_from(edges)
    graph.layout(prog="dot")

    # Save the graph as an image
    graph.draw("best_tree.png")

    # Display the graph using PIL
    image = Image.open("best_tree.png")
    image.show()


if __name__ == "__main__":
    main()
    