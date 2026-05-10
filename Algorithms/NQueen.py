import random
import matplotlib.pyplot as plt


class NQueens:

    def __init__(
        self,
        n=8,
        population_size=100,
        generations=200,
        mutation_rate=0.01
    ):

        self.n = n

        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate

        self.population = []
        self.history = []

    def create_individual(self):

        individual = list(range(self.n))

        random.shuffle(individual)

        return individual

    def create_population(self):

        return [

            self.create_individual()

            for _ in range(self.population_size)
        ]

    def fitness(self, individual):

        conflicts = 0

        for i in range(self.n):

            for j in range(i + 1, self.n):

                if abs(
                    individual[i] - individual[j]
                ) == abs(i - j):

                    conflicts += 1

        return -conflicts

    def selection(self, population, k=3):

        selected = random.sample(population, k)

        return max(selected, key=self.fitness)

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

    def mutate(self, individual):

        if random.random() < self.mutation_rate:

            i, j = random.sample(
                range(self.n), 2
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

            best = max(population, key=self.fitness)

            new_population.append(best)

            while len(new_population) < self.population_size:

                p1 = self.selection(population)
                p2 = self.selection(population)

                child = self.crossover(p1, p2)

                child = self.mutate(child)

                new_population.append(child)

            population = new_population

            best = max(population, key=self.fitness)

            best_conflicts = -self.fitness(best)

            self.history.append(best_conflicts)

            print(
                f"Gen {gen}: "
                f"Conflicts = {best_conflicts}"
            )

            best_solution = best

            if best_conflicts == 0:
                break

        return best_solution

    def plot_board(self, solution):

        plt.figure(figsize=(5, 5))

        for i in range(self.n):

            for j in range(self.n):

                color = (
                    'white'
                    if (i + j) % 2 == 0
                    else 'gray'
                )

                plt.gca().add_patch(

                    plt.Rectangle(
                        (j, i),
                        1,
                        1,
                        color=color
                    )
                )

        for row in range(self.n):

            col = solution[row]

            plt.text(
                col + 0.5,
                row + 0.5,
                'Q',
                ha='center',
                va='center',
                fontsize=16,
                color='red'
            )

        plt.xlim(0, self.n)
        plt.ylim(0, self.n)

        plt.gca().invert_yaxis()

        plt.title("N-Queens Solution")

        plt.show()

    def plot_history(self):

        plt.plot(self.history)

        plt.title(
            "Conflicts over Generations"
        )

        plt.xlabel("Generation")

        plt.ylabel("Conflicts")

        plt.grid()

        plt.show()