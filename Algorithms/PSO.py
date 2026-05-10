import numpy as np
import math
import matplotlib.pyplot as plt


class PSO:

    def __init__(
        self,
        dimensions=2,
        population=30,
        max_iter=100,
        min_bound=-5,
        max_bound=5,
        w=0.7,
        c1=1.5,
        c2=1.5,
        v_max=0.5
    ):

        self.dimensions = dimensions

        self.population = population

        self.max_iter = max_iter

        self.min_bound = min_bound
        self.max_bound = max_bound

        self.w = w
        self.c1 = c1
        self.c2 = c2

        self.v_max = v_max

        self.swarm = []

        self.history = []

    def cost_function(self, x):

        a = 20
        b = 0.2
        c = 2 * math.pi

        d = len(x)

        term1 = -a * np.exp(
            -b * np.sqrt(np.sum(x**2) / d)
        )

        term2 = -np.exp(
            np.sum(np.cos(c * x)) / d
        )

        return term1 + term2 + a + math.e

    class Particle:

        def __init__(
            self,
            dimensions,
            min_bound,
            max_bound,
            v_max,
            cost_function
        ):

            self.position = np.random.uniform(
                min_bound,
                max_bound,
                dimensions
            )

            self.velocity = np.random.uniform(
                -v_max,
                v_max,
                dimensions
            )

            self.best_position = (
                self.position.copy()
            )

            self.fitness = cost_function(
                self.position
            )

            self.best_fitness = self.fitness

    def initialize_swarm(self):

        self.swarm = [

            self.Particle(
                self.dimensions,
                self.min_bound,
                self.max_bound,
                self.v_max,
                self.cost_function
            )

            for _ in range(self.population)
        ]

    def run(self):

        self.initialize_swarm()

        gbest_particle = min(
            self.swarm,
            key=lambda p: p.best_fitness
        )

        gbest_position = (
            gbest_particle.best_position.copy()
        )

        gbest_fitness = (
            gbest_particle.best_fitness
        )

        for iteration in range(self.max_iter):

            for particle in self.swarm:

                r1 = np.random.rand(
                    self.dimensions
                )

                r2 = np.random.rand(
                    self.dimensions
                )

                particle.velocity = (

                    self.w * particle.velocity

                    +

                    self.c1 * r1 * (
                        particle.best_position
                        - particle.position
                    )

                    +

                    self.c2 * r2 * (
                        gbest_position
                        - particle.position
                    )
                )

                particle.velocity = np.clip(
                    particle.velocity,
                    -self.v_max,
                    self.v_max
                )

                particle.position = (
                    particle.position
                    + particle.velocity
                )

                particle.position = np.clip(
                    particle.position,
                    self.min_bound,
                    self.max_bound
                )

                particle.fitness = (
                    self.cost_function(
                        particle.position
                    )
                )

                if particle.fitness < particle.best_fitness:

                    particle.best_fitness = (
                        particle.fitness
                    )

                    particle.best_position = (
                        particle.position.copy()
                    )

                if particle.fitness < gbest_fitness:

                    gbest_fitness = (
                        particle.fitness
                    )

                    gbest_position = (
                        particle.position.copy()
                    )

            self.history.append(gbest_fitness)

            print(
                f"Iteration "
                f"{iteration+1}/{self.max_iter} "
                f"| Best Fitness = "
                f"{gbest_fitness:.6f}"
            )

        return gbest_position, gbest_fitness

    def plot_history(self):

        plt.plot(self.history)

        plt.title(
            "PSO Convergence Curve"
        )

        plt.xlabel("Iteration")

        plt.ylabel("Best Fitness")

        plt.grid()

        plt.show()