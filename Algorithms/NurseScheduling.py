import random
import numpy as np
import matplotlib.pyplot as plt

from .GeneticAlgorithmBase import GeneticAlgorithmBase


class NurseScheduling(GeneticAlgorithmBase):

    def __init__(
        self,
        num_nurses=4,
        days=5,
        shifts=[0, 1, 2, 3],
        max_shifts=4,
        population_size=20,
        generations=50,
        mutation_rate=0.3
    ):
        super().__init__(population_size, generations, mutation_rate)

        self.num_nurses = num_nurses
        self.days = days

        self.shifts = shifts
        self.max_shifts = max_shifts

    def create_individual(self):

        return np.random.choice(
            self.shifts,
            size=(self.num_nurses, self.days)
        )

    def create_population(self):

        return [

            self.create_individual()

            for _ in range(self.population_size)
        ]

    def fitness(self, individual):
        penalty = 0

        # 1. Check max shifts per nurse (Your original rule)
        for nurse in individual:
            work_days = np.count_nonzero(nurse)
            if work_days > self.max_shifts:
                penalty += (work_days - self.max_shifts) * 2

        # 2. Check strict shift coverage per day
        for day in range(self.days):
            daily_shifts = individual[:, day]
            
            # Penalize if a specific shift is missing completely
            if 1 not in daily_shifts:  # No Morning nurse assigned
                penalty += 5
            if 2 not in daily_shifts:  # No Evening nurse assigned
                penalty += 5
            if 3 not in daily_shifts:  # No Night nurse assigned
                penalty += 5
                
            # Penalize if NO ONE is working at all (Your original rule, made stricter)
            if np.all(daily_shifts == 0):
                penalty += 10

        return -penalty
    def selection(self, population):

        return max(
            random.sample(population, 3),
            key=self.fitness
        )

    def crossover(self, p1, p2):

        point = random.randint(
            1,
            self.days - 1
        )

        child = np.hstack(
            (
                p1[:, :point],
                p2[:, point:]
            )
        )

        return child

    def mutation(self, individual):

        if random.random() < self.mutation_rate:

            nurse = random.randint(
                0,
                self.num_nurses - 1
            )

            day = random.randint(
                0,
                self.days - 1
            )

            individual[nurse][day] = random.choice(
                self.shifts
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

            best_fit = self.fitness(
                best_solution
            )

            self.history.append(best_fit)

            print(
                f"Generation {gen}: "
                f"Fitness = {best_fit}"
            )

        return best_solution

    def plot_schedule(self, schedule):

        plt.imshow(schedule, aspect='auto')

        plt.title(
            "Nurse Scheduling Heatmap"
        )

        plt.xlabel("Days")

        plt.ylabel("Nurses")

        plt.colorbar(
            label="Shift "
            "(0=Off,1=M,2=E,3=N)"
        )

        plt.show()

    def plot_history(self):

        plt.plot(self.history, marker='o')

        plt.title(
            "Fitness Over Generations"
        )

        plt.xlabel("Generation")

        plt.ylabel("Fitness")

        plt.grid()

        plt.show()

    def plot_best_generation(self):

        best_gen = self.history.index(
            max(self.history)
        )

        plt.plot(self.history)

        plt.axvline(
            x=best_gen,
            linestyle='--'
        )

        plt.title("Best Solution Point")

        plt.show()

        print(
            "Best solution found at generation:",
            best_gen
        )