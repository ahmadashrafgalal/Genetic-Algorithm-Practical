from abc import ABC, abstractmethod


class GeneticAlgorithmBase(ABC):
    """
    Base class for all Genetic Algorithm implementations.
    Provides common interface for GA operations.
    """

    def __init__(self, population_size=20, generations=50, mutation_rate=0.1):
        """
        Initialize base GA parameters.
        
        Args:
            population_size (int): Size of the population
            generations (int): Number of generations to evolve
            mutation_rate (float): Probability of mutation
        """
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.population = []
        self.history = []

    @abstractmethod
    def fitness(self, individual):
        """
        Calculate fitness score for an individual.
        
        Args:
            individual: The individual to evaluate
            
        Returns:
            float: Fitness score
        """
        pass

    @abstractmethod
    def crossover(self, parent1, parent2):
        """
        Perform crossover between two parents to create offspring.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            child: Offspring from parents
        """
        pass

    @abstractmethod
    def mutation(self, individual):
        """
        Apply mutation to an individual.
        
        Args:
            individual: The individual to mutate
            
        Returns:
            individual: Mutated individual (or None if mutation is in-place)
        """
        pass
