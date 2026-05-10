import numpy as np
import random
import matplotlib.pyplot as plt


class TSP:

    def __init__(
        self,
        cities_coords,
        city_labels,
        population_size=30,
        generations=40,
        mutation_rate=0.25,
        elite_size=1,
        seed=42
    ):

        np.random.seed(seed)
        random.seed(seed)

        self.cities_coords = np.array(cities_coords)
        self.city_labels = city_labels
        self.n_cities = len(cities_coords)

        self.dist_matrix = np.linalg.norm(
            self.cities_coords[:, None, :] -
            self.cities_coords[None, :, :],
            axis=2
        )

        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size

        self.population = []
        self.history = []

        self.best_cost_overall = float('inf')
        self.best_chrom_overall = None

    def normalize(self, chrom):
        idx = chrom.index(0)
        return chrom[idx:] + chrom[:idx]

    def create_chromosome(self):
        chrom = list(range(self.n_cities))
        random.shuffle(chrom)
        return self.normalize(chrom)

    def decode(self, chrom):
        return [self.city_labels[i] for i in chrom] + [
            self.city_labels[chrom[0]]
        ]

    def cost(self, chrom):
        total = 0

        for i in range(self.n_cities):
            total += self.dist_matrix[
                chrom[i],
                chrom[(i + 1) % self.n_cities]
            ]

        return total

    def fitness(self, costs):
        max_cost = max(costs)
        return [max_cost - c + 1e-6 for c in costs]

    def selection(self, population, fitnesses, k=3):

        selected = random.sample(
            range(len(population)), k
        )

        best = max(
            selected,
            key=lambda i: fitnesses[i]
        )

        return population[best][:]

    def crossover(self, p1, p2):

        size = len(p1)
        start, end = sorted(
            random.sample(range(size), 2)
        )

        child = [None] * size
        child[start:end] = p1[start:end]

        idx = end % size

        for gene in p2:
            if gene not in child:

                while child[idx] is not None:
                    idx = (idx + 1) % size

                child[idx] = gene
                idx = (idx + 1) % size

        return self.normalize(child)

    def mutate(self, chrom):

        if random.random() < self.mutation_rate:

            i, j = random.sample(
                range(self.n_cities), 2
            )

            chrom[i], chrom[j] = chrom[j], chrom[i]

        return self.normalize(chrom)

    def init_population(self):

        self.population = [
            self.create_chromosome()
            for _ in range(self.population_size)
        ]

    def evolve_one_generation(self):

        costs = [
            self.cost(ch)
            for ch in self.population
        ]

        fitnesses = self.fitness(costs)

        best_idx = np.argmax(fitnesses)

        best_cost = costs[best_idx]
        best_chrom = self.population[best_idx]

        if best_cost < self.best_cost_overall:
            self.best_cost_overall = best_cost
            self.best_chrom_overall = best_chrom[:]

        self.history.append(self.best_cost_overall)

        new_population = []

        new_population.append(best_chrom[:])

        while len(new_population) < self.population_size:

            p1 = self.selection(
                self.population, fitnesses
            )

            p2 = self.selection(
                self.population, fitnesses
            )

            child = self.crossover(p1, p2)
            child = self.mutate(child)

            new_population.append(child)

        self.population = new_population

    def run(self):

        self.init_population()

        for _ in range(self.generations):

            self.evolve_one_generation()

        return self.best_chrom_overall

    def get_result(self):

        return {
            "tour": self.decode(self.best_chrom_overall),
            "cost": self.best_cost_overall
        }

    def plot_tour(self, chrom):

        plt.figure(figsize=(7, 5))

        plt.scatter(
            self.cities_coords[:, 0],
            self.cities_coords[:, 1],
            s=150
        )

        for i, label in enumerate(self.city_labels):

            plt.text(
                self.cities_coords[i, 0] + 0.1,
                self.cities_coords[i, 1] + 0.1,
                label
            )

        route = chrom + [chrom[0]]

        for i in range(len(route) - 1):

            p1 = self.cities_coords[route[i]]
            p2 = self.cities_coords[route[i + 1]]

            plt.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]]
            )

        plt.grid(True)
        plt.show()

    def plot_convergence(self):

        plt.figure()
        plt.plot(self.history)
        plt.title("GA Convergence")
        plt.xlabel("Generation")
        plt.ylabel("Best Cost")
        plt.grid(True)
        plt.show()