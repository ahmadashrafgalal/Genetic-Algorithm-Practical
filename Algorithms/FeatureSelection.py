import random
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    r2_score
)

from .GeneticAlgorithmBase import GeneticAlgorithmBase


class FeatureSelection(GeneticAlgorithmBase):

    def __init__(
        self,
        X,
        y,
        model,
        task="classification",
        population_size=20,
        generations=30,
        mutation_rate=0.3,
        test_size=0.3
    ):
        super().__init__(population_size, generations, mutation_rate)

        self.X = X
        self.y = y

        self.X_train, self.X_test, \
        self.y_train, self.y_test = train_test_split(
            X,
            y,
            test_size=test_size
        )

        self.model = model

        self.task = task

        self.num_features = X.shape[1]

    def create_individual(self):

        return [

            random.randint(0, 1)

            for _ in range(self.num_features)
        ]

    def create_population(self):

        return [

            self.create_individual()

            for _ in range(self.population_size)
        ]

    def fitness(self, individual):

        if sum(individual) == 0:

            return -1

        selected = [

            i

            for i in range(self.num_features)

            if individual[i] == 1
        ]

        model = self.model

        model.fit(
            self.X_train[:, selected],
            self.y_train
        )

        preds = model.predict(
            self.X_test[:, selected]
        )

        if self.task == "classification":

            return accuracy_score(
                self.y_test,
                preds
            )

        elif self.task == "regression":

            return r2_score(
                self.y_test,
                preds
            )

    def selection(self, population):

        return max(
            random.sample(population, 3),
            key=self.fitness
        )

    def crossover(self, p1, p2):

        point = random.randint(
            1,
            self.num_features - 1
        )

        return (
            p1[:point] +
            p2[point:]
        )

    def mutation(self, individual):

        if random.random() < self.mutation_rate:

            i = random.randint(
                0,
                self.num_features - 1
            )

            individual[i] = 1 - individual[i]

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
                f"Gen {gen}: "
                f"Fitness = {best_fit:.4f}"
            )

        return best_solution

    def get_selected_features(self, mask):

        return [

            i

            for i in range(self.num_features)

            if mask[i] == 1
        ]

    def plot_features(self, mask):

        plt.figure(figsize=(10, 4))

        plt.bar(

            [f"F{i}" for i in range(len(mask))],

            mask
        )

        plt.title("Selected Features")

        plt.ylim(0, 1)

        plt.show()

    def plot_history(self):

        plt.plot(self.history)

        plt.title(
            "Fitness Over Generations"
        )

        plt.xlabel("Generation")

        plt.ylabel("Fitness")

        plt.grid()

        plt.show()