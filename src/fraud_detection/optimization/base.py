"""
Base Metaheuristic Module
Abstract base class for binary metaheuristic algorithms.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Tuple, List
from ..config import META_PARAMS

class BinaryMetaheuristic(ABC):
    """Abstract base class for binary metaheuristic algorithms."""
    
    def __init__(self, n_features: int, population_size: int = None, max_iterations: int = None):
        """
        Initialize metaheuristic.
        """
        self.n_features = n_features
        self.population_size = population_size or META_PARAMS['population_size']
        self.max_iterations = max_iterations or META_PARAMS['max_iterations']
        self.fitness_weight = META_PARAMS['fitness_weight']
        self.feature_weight = META_PARAMS['feature_weight']
        
        # Initialize population
        self.population = np.random.randint(0, 2, (self.population_size, n_features))
        self.fitness = np.zeros(self.population_size)
        self.best_solution = None
        self.best_fitness = -np.inf
        self.convergence_curve = []
        
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        
    def binary_conversion(self, continuous_values: np.ndarray) -> np.ndarray:
        """Convert continuous values to binary using sigmoid probability."""
        probs = self.sigmoid(continuous_values)
        return (np.random.rand(*probs.shape) < probs).astype(int)
        
    def calculate_fitness(self, solution: np.ndarray, fitness_func: Callable) -> float:
        """Calculate weighted fitness."""
        n_selected = np.sum(solution)
        if n_selected == 0:
            return 0.0
            
        accuracy = fitness_func(solution)
        
        # Weighted sum of accuracy and feature reduction
        fitness = (self.fitness_weight * accuracy - 
                  self.feature_weight * (n_selected / self.n_features))
                  
        return fitness
        
    def evaluate_population(self, fitness_func: Callable):
        """Evaluate entire population."""
        for i in range(self.population_size):
            self.fitness[i] = self.calculate_fitness(self.population[i], fitness_func)
            
            if self.fitness[i] > self.best_fitness:
                self.best_fitness = self.fitness[i]
                self.best_solution = self.population[i].copy()
                
    @abstractmethod
    def update_population(self, iteration: int):
        """Update population positions (implement in subclass)."""
        pass
        
    def optimize(self, fitness_func: Callable) -> Tuple[np.ndarray, float, List[float]]:
        """Run the optimization loop."""
        self.evaluate_population(fitness_func)
        self.convergence_curve.append(self.best_fitness)
        
        for iteration in range(self.max_iterations):
            self.update_population(iteration)
            self.evaluate_population(fitness_func)
            self.convergence_curve.append(self.best_fitness)
            
        return self.best_solution, self.best_fitness, self.convergence_curve
