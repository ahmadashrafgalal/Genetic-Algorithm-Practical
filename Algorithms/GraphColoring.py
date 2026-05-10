import random
import networkx as nx
import matplotlib.pyplot as plt


class GraphColoring:

    def __init__(
        self,
        graph,
        num_colors=3,
        population_size=20,
        generations=50,
        mutation_rate=0.3
    ):

        self.G = graph

        self.num_nodes = self.G.number_of_nodes()

        self.num_colors = num_colors

        self.population_size = population_size

        self.generations = generations

        self.mutation_rate = mutation_rate

        self.population = []

        self.history = []

    def create_individual(self):

        return [

            random.randint(
                0,
                self.num_colors - 1
            )

            for _ in range(self.num_nodes)
        ]

    def create_population(self):

        return [

            self.create_individual()

            for _ in range(self.population_size)
        ]

    def fitness(self, individual):

        conflicts = 0

        for u, v in self.G.edges():

            if individual[u] == individual[v]:

                conflicts += 1

        return -conflicts

    def selection(self, population):

        return max(
            random.sample(population, 3),
            key=self.fitness
        )

    def crossover(self, p1, p2):

        point = random.randint(
            1,
            self.num_nodes - 1
        )

        child = (
            p1[:point] +
            p2[point:]
        )

        return child

    def mutate(self, individual):

        if random.random() < self.mutation_rate:

            i = random.randint(
                0,
                self.num_nodes - 1
            )

            individual[i] = random.randint(
                0,
                self.num_colors - 1
            )

        return individual

    def run(self):

        population = self.create_population()

        best_solution = None

        for gen in range(self.generations):

            new_population = []

            for _ in range(self.population_size):

                p1 = self.selection(population)

                p2 = self.selection(population)

                child = self.crossover(p1, p2)

                child = self.mutate(child)

                new_population.append(child)

            population = new_population

            best_solution = max(
                population,
                key=self.fitness
            )

            best_fit = self.fitness(
                best_solution
            )

            self.history.append(best_fit)

            print(
                f"Gen {gen}: "
                f"Conflicts = {-best_fit}"
            )

            if best_fit == 0:
                break

        return best_solution

    def visualize(self, solution):

        pos = nx.spring_layout(self.G)

        nx.draw(

            self.G,
            pos,

            node_color=solution,

            with_labels=True,

            cmap=plt.cm.Set3,

            node_size=800
        )

        plt.title("Graph Coloring Result")

        plt.show()

    def plot_history(self):

        plt.plot(self.history)

        plt.title(
            "Conflicts Over Generations"
        )

        plt.xlabel("Generation")

        plt.ylabel("Conflicts")

        plt.grid()

        plt.show()