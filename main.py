from Algorithms.KnapSack import Knapsack
from Algorithms.PSO import PSO
from Algorithms.TSP import TSP 
from Algorithms.VRP import VRP 
from Algorithms.NQueen import NQueens
from Algorithms.NurseScheduling import NurseScheduling 
from Algorithms.GraphColoring import GraphColoring 
from Algorithms.FeatureSelection import FeatureSelection

import networkx as nx
from sklearn.datasets import make_friedman1
from sklearn.linear_model import LinearRegression



print("\n" + "="*50 + "\n")
print("Knapsack Solution:")

items = ['A', 'B', 'C', 'D', 'E']
values = [50, 30, 20, 30, 50]
weights = [30, 50, 40, 20, 60]

ga = Knapsack(
    items=items,
    values=values,
    weights=weights,
    capacity=110,
    population_size=6,
    generations=30,
    mutation_rate=0.1
)

best_solution = ga.run()

ga.print_results(best_solution)

# ga.plot_fitness()
# ga.plot_solution(best_solution)


print("\n" + "="*50 + "\n")
print("TSP Solution:")

tsp = TSP(
    cities_coords=[
        [0, 0],
        [1, 3],
        [4, 2],
        [3, 0],
        [1, 4]
    ],
    city_labels=['A', 'B', 'C', 'D', 'E']
)

best = tsp.run()

result = tsp.get_result()

print(result["tour"])
print(result["cost"])

# tsp.plot_tour(best)
# tsp.plot_convergence()


print("\n" + "="*50 + "\n")
print("VRP Solution:")

depot = (0, 0)
customers = {
    1: (2, 3),
    2: (5, 4),
    3: (1, 7),
    4: (6, 8),
    5: (3, 6)
}

vrp = VRP(
    depot=depot,
    customers=customers,
    num_vehicles=2,
    population_size=20,
    generations=100,
    mutation_rate=0.1
)

best = vrp.run()

routes = vrp.get_routes(best)

print("\nBest Routes:")
for i, r in enumerate(routes):
    print(f"Vehicle {i+1}: {r}")

# vrp.plot(routes)
# vrp.plot_convergence()


print("\n" + "="*50 + "\n")
print("N-Queens Solution:")

queens = NQueens(
    n=8,
    population_size=100,
    generations=200,
    mutation_rate=0.01
)

best_solution = queens.run()

print("\nBest Solution:", best_solution)

print(
    "Final Conflicts:",
    -queens.fitness(best_solution)
)

# queens.plot_board(best_solution)

# queens.plot_history()



print("\n" + "="*50 + "\n")
print("Nurse Scheduling Solution:")

scheduler = NurseScheduling(
    num_nurses=4,
    days=5,
    population_size=20,
    generations=50
)

best_schedule = scheduler.run()

print("\nBest Schedule:\n")
print(best_schedule)

# scheduler.plot_schedule(best_schedule)

# scheduler.plot_history()

# scheduler.plot_best_generation()


print("\n" + "="*50 + "\n")
print("Color Graphing Solution:")

G = nx.Graph()

edges = [
    (0,1),(0,2),(1,2),(1,3),
    (2,3),(3,4),(4,5),(5,0)
]

G.add_edges_from(edges)

gc = GraphColoring(
    graph=G,
    num_colors=3,
    population_size=20,
    generations=50
)

best_solution = gc.run()

print("\nBest Coloring:")
print(best_solution)

# gc.visualize(best_solution)

# gc.plot_history()

print("\n" + "="*50 + "\n")
print("Feature Selection Solution:")


X, y = make_friedman1(
    n_samples=200,
    n_features=10,
    noise=0.1
)

ga = FeatureSelection(
    X=X,
    y=y,
    model=LinearRegression(),
    task="regression"
)

best = ga.run()

print(
    "Selected Features:",
    ga.get_selected_features(best)
)

ga.plot_features(best)

ga.plot_history()


print("\n" + "="*50 + "\n")
print("PSO Solution:")

pso = PSO()

best_position, best_fitness = pso.run()

print("\nOptimal Solution:")
print("Best Position:", best_position)
print("Best Fitness:", best_fitness)

pso.plot_history()