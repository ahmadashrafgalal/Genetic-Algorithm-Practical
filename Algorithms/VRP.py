import random
import math
import matplotlib.pyplot as plt

from .GeneticAlgorithmBase import GeneticAlgorithmBase


class VRP(GeneticAlgorithmBase):

    def __init__(
        self,
        depot,
        customers,
        num_vehicles,
        population_size=20,
        generations=100,
        mutation_rate=0.1
    ):
        super().__init__(population_size, generations, mutation_rate)

        self.depot = depot
        self.customers = customers
        self.customer_ids = list(customers.keys())
        self.num_vehicles = num_vehicles

    def distance(self, p1, p2):

        return math.sqrt(
            (p1[0] - p2[0])**2 +
            (p1[1] - p2[1])**2
        )

    def decode(self, chromosome):

        routes = [[] for _ in range(self.num_vehicles)]

        for i, cust in enumerate(chromosome):

            routes[i % self.num_vehicles].append(cust)

        return routes

    def fitness(self, chromosome):

        routes = self.decode(chromosome)

        total_distance = 0

        for route in routes:

            if not route:
                continue

            prev = self.depot

            for c in route:

                total_distance += self.distance(
                    prev,
                    self.customers[c]
                )

                prev = self.customers[c]

            total_distance += self.distance(
                prev,
                self.depot
            )

        return 1 / total_distance

    def create_individual(self):

        indiv = self.customer_ids[:]
        random.shuffle(indiv)

        return indiv

    def create_population(self):

        return [
            self.create_individual()
            for _ in range(self.population_size)
        ]

    def selection(self, population):

        return random.choice(population)

    def crossover(self, p1, p2):

        size = len(p1)
        start, end = sorted(
            random.sample(range(size), 2)
        )

        child = [None] * size
        child[start:end] = p1[start:end]

        fill = [
            x for x in p2
            if x not in child
        ]

        j = 0

        for i in range(size):

            if child[i] is None:
                child[i] = fill[j]
                j += 1

        return child

    def mutation(self, individual):

        if random.random() < self.mutation_rate:

            i, j = random.sample(
                range(len(individual)), 2
            )

            individual[i], individual[j] = (
                individual[j],
                individual[i]
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
                child = self.mutation(child)

                new_population.append(child)

            population = new_population

            best_solution = max(
                population,
                key=self.fitness
            )

            best_distance = 1 / self.fitness(
                best_solution
            )

            self.history.append(best_distance)

            print(
                f"Gen {gen}: "
                f"Best Distance = {best_distance:.2f}"
            )

        return best_solution

    def get_routes(self, chromosome):

        return self.decode(chromosome)

    def plot(self, routes):

        plt.figure()

        plt.scatter(
            self.depot[0],
            self.depot[1]
        )

        plt.text(
            self.depot[0],
            self.depot[1],
            "Depot"
        )

        for cid, (x, y) in self.customers.items():

            plt.scatter(x, y)
            plt.text(x, y, f"C{cid}")

        for route in routes:

            x_coords = [self.depot[0]]
            y_coords = [self.depot[1]]

            for c in route:

                x_coords.append(
                    self.customers[c][0]
                )

                y_coords.append(
                    self.customers[c][1]
                )

            x_coords.append(self.depot[0])
            y_coords.append(self.depot[1])

            plt.plot(x_coords, y_coords)

        plt.title("VRP Routes")
        plt.grid()
        plt.show()

    def plot_convergence(self):

        plt.figure()
        plt.plot(self.history)
        plt.title("VRP Convergence")
        plt.xlabel("Generation")
        plt.ylabel("Distance")
        plt.grid()
        plt.show()