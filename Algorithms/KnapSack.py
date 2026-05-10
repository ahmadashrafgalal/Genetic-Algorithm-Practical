import random
import matplotlib.pyplot as plt


class Knapsack:

    def __init__(
        self,
        items,
        values,
        weights,
        capacity,
        population_size=6,
        generations=30,
        mutation_rate=0.1
    ):

        self.items = items
        self.values = values
        self.weights = weights
        self.capacity = capacity

        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate

        self.num_items = len(items)

        self.population = []
        self.best_fitness_history = []


    def create_population(self):

        population = []

        for _ in range(self.population_size):

            chromosome = [
                random.randint(0, 1)
                for _ in range(self.num_items)
            ]

            population.append(chromosome)

        return population


    def fitness(self, chromosome):

        total_value = 0
        total_weight = 0

        for i in range(self.num_items):

            if chromosome[i] == 1:

                total_value += self.values[i]
                total_weight += self.weights[i]

        if total_weight > self.capacity:
            return 0

        return total_value


    def selection(self):

        fitness_values = [
            self.fitness(ch)
            for ch in self.population
        ]

        total_fitness = sum(fitness_values)

        if total_fitness == 0:
            return random.choice(self.population)

        probabilities = [
            f / total_fitness
            for f in fitness_values
        ]

        selected = random.choices(
            self.population,
            probabilities
        )[0]

        return selected


    def crossover(self, parent1, parent2):

        point = random.randint(1, self.num_items - 1)

        child = (
            parent1[:point] +
            parent2[point:]
        )

        return child


    def mutation(self, chromosome):

        for i in range(self.num_items):

            if random.random() < self.mutation_rate:

                chromosome[i] = 1 - chromosome[i]

        return chromosome


    def run(self):

        self.population = self.create_population()

        for generation in range(self.generations):

            new_population = []

            for _ in range(self.population_size):

                parent1 = self.selection()
                parent2 = self.selection()

                child = self.crossover(parent1, parent2)

                child = self.mutation(child)

                new_population.append(child)

            self.population = new_population

            best = max(
                self.population,
                key=self.fitness
            )

            self.best_fitness_history.append(
                self.fitness(best)
            )

        best_solution = max(
            self.population,
            key=self.fitness
        )

        return best_solution


    def print_results(self, best_solution):

        print("Best Chromosome:", best_solution)

        selected_items = [

            self.items[i]

            for i in range(self.num_items)

            if best_solution[i] == 1
        ]

        print("Selected Items:", selected_items)

        print(
            "Best Fitness:",
            self.fitness(best_solution)
        )

        total_weight = sum(

            self.weights[i]

            for i in range(self.num_items)

            if best_solution[i] == 1
        )

        print("Total Weight:", total_weight)


    def plot_fitness(self):

        plt.figure()

        plt.plot(self.best_fitness_history)

        plt.title(
            "Fitness Improvement Over Generations"
        )

        plt.xlabel("Generation")

        plt.ylabel("Best Fitness")

        plt.show()


    def plot_solution(self, best_solution):

        selected = [
            best_solution[i]
            for i in range(self.num_items)
        ]

        plt.figure()

        plt.bar(self.items, selected)

        plt.title("Items Selected")

        plt.xlabel("Items")

        plt.ylabel("Selected")

        plt.show()