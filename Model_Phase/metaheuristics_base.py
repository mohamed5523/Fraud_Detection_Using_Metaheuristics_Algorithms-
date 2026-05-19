"""
Base Metaheuristic Algorithm Class
Provides foundation for all binary metaheuristic implementations
"""

import numpy as np
from abc import ABC, abstractmethod
import config


class BinaryMetaheuristic(ABC):
    """Abstract base class for binary metaheuristic algorithms"""
    
    def __init__(self, n_features, population_size=None, max_iterations=None):
        """
        Initialize metaheuristic algorithm
        
        Args:
            n_features: Number of features in dataset
            population_size: Size of population
            max_iterations: Maximum number of iterations
        """
        self.n_features = n_features
        self.population_size = population_size or config.META_PARAMS['population_size']
        self.max_iterations = max_iterations or config.META_PARAMS['max_iterations']
        self.fitness_weight = config.META_PARAMS['fitness_weight']
        self.feature_weight = config.META_PARAMS['feature_weight']
        
        # Initialize population randomly
        self.population = np.random.randint(0, 2, (self.population_size, n_features))
        self.fitness = np.zeros(self.population_size)
        self.best_solution = None
        self.best_fitness = -np.inf
        self.convergence_curve = []
        
    def sigmoid(self, x):
        """Sigmoid transfer function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def tanh_transfer(self, x):
        """Tanh transfer function"""
        return np.abs(np.tanh(x))
    
    def binary_conversion(self, continuous_values):
        """Convert continuous values to binary using sigmoid"""
        probs = self.sigmoid(continuous_values)
        return (np.random.rand(*probs.shape) < probs).astype(int)
    
    def calculate_fitness(self, solution, fitness_func):
        """
        Calculate fitness for a solution
        
        Args:
            solution: Binary array representing feature selection
            fitness_func: Function to evaluate solution
            
        Returns:
            Fitness value
        """
        n_selected = np.sum(solution)
        
        # Ensure at least one feature is selected
        if n_selected == 0:
            return 0.0
        
        # Get accuracy from fitness function
        accuracy = fitness_func(solution)
        
        # Fitness = accuracy - penalty for number of features
        fitness = (self.fitness_weight * accuracy - 
                  self.feature_weight * (n_selected / self.n_features))
        
        return fitness
    
    def evaluate_population(self, fitness_func):
        """Evaluate fitness for entire population"""
        for i in range(self.population_size):
            self.fitness[i] = self.calculate_fitness(self.population[i], fitness_func)
            
            # Update best solution
            if self.fitness[i] > self.best_fitness:
                self.best_fitness = self.fitness[i]
                self.best_solution = self.population[i].copy()
    
    @abstractmethod
    def update_population(self, iteration):
        """
        Update population (must be implemented by subclasses)
        
        Args:
            iteration: Current iteration number
        """
        pass
    
    def optimize(self, fitness_func):
        """
        Run optimization process
        
        Args:
            fitness_func: Function to evaluate solutions
            
        Returns:
            best_solution, best_fitness, convergence_curve
        """
        # Initial evaluation
        self.evaluate_population(fitness_func)
        self.convergence_curve.append(self.best_fitness)
        
        # Main optimization loop
        for iteration in range(self.max_iterations):
            # Update population (algorithm-specific)
            self.update_population(iteration)
            
            # Evaluate new population
            self.evaluate_population(fitness_func)
            
            # Store convergence
            self.convergence_curve.append(self.best_fitness)
        
        return self.best_solution, self.best_fitness, self.convergence_curve
    
    def get_selected_features(self):
        """Get indices of selected features"""
        return np.where(self.best_solution == 1)[0]


class TransferFunction:
    """Transfer functions for binary conversion"""
    
    @staticmethod
    def s_shaped_1(x):
        """S-shaped transfer function 1"""
        return 1 / (1 + np.exp(-2 * x))
    
    @staticmethod
    def s_shaped_2(x):
        """S-shaped transfer function 2"""
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def s_shaped_3(x):
        """S-shaped transfer function 3"""
        return 1 / (1 + np.exp(-x / 2))
    
    @staticmethod
    def s_shaped_4(x):
        """S-shaped transfer function 4"""
        return 1 / (1 + np.exp(-x / 3))
    
    @staticmethod
    def v_shaped_1(x):
        """V-shaped transfer function 1"""
        return np.abs(np.erf((np.sqrt(np.pi) / 2) * x))
    
    @staticmethod
    def v_shaped_2(x):
        """V-shaped transfer function 2"""
        return np.abs(np.tanh(x))
    
    @staticmethod
    def v_shaped_3(x):
        """V-shaped transfer function 3"""
        return np.abs(x / np.sqrt(1 + x**2))
    
    @staticmethod
    def v_shaped_4(x):
        """V-shaped transfer function 4"""
        return np.abs((2 / np.pi) * np.arctan((np.pi / 2) * x))
