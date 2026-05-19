"""
Implementation of 11 Binary Metaheuristic Algorithms
"""

import numpy as np
from metaheuristics_base import BinaryMetaheuristic


class BHHO(BinaryMetaheuristic):
    """Binary Harris Hawks Optimization"""
    
    def update_population(self, iteration):
        E1 = 2 * (1 - iteration / self.max_iterations)
        
        for i in range(self.population_size):
            E0 = 2 * np.random.rand() - 1
            E = E1 * E0
            
            if np.abs(E) >= 1:
                # Exploration
                rand_idx = np.random.randint(0, self.population_size)
                r1, r2 = np.random.rand(2)
                X_new = self.population[rand_idx] - r1 * np.abs(
                    self.population[rand_idx] - 2 * r2 * self.population[i]
                )
            else:
                # Exploitation
                r = np.random.rand()
                if r >= 0.5 and np.abs(E) >= 0.5:
                    X_new = self.best_solution - E * np.abs(
                        self.best_solution - self.population[i]
                    )
                else:
                    X_new = self.best_solution - self.population[i]
            
            self.population[i] = self.binary_conversion(X_new)


class BHGSO(BinaryMetaheuristic):
    """Binary Henry Gas Solubility Optimization"""
    
    def update_population(self, iteration):
        T = np.exp(-iteration / self.max_iterations)
        K = np.random.rand()
        
        for i in range(self.population_size):
            # Henry's coefficient
            H = K * T
            
            # Partial pressure
            P = self.fitness[i] / (np.sum(self.fitness) + 1e-10)
            
            # Concentration
            C = H * P
            
            # Update position
            X_new = self.population[i] + C * np.random.randn(self.n_features)
            self.population[i] = self.binary_conversion(X_new)


class BASO(BinaryMetaheuristic):
    """Binary Atom Search Optimization"""
    
    def update_population(self, iteration):
        alpha = 50
        beta = 0.2
        
        # Calculate mass
        mass = np.exp(-(self.fitness - np.max(self.fitness)) / 
                      (np.max(self.fitness) - np.min(self.fitness) + 1e-10))
        mass = mass / (np.sum(mass) + 1e-10)
        
        for i in range(self.population_size):
            # Calculate acceleration
            acceleration = np.zeros(self.n_features)
            
            for j in range(self.population_size):
                if i != j:
                    distance = np.linalg.norm(self.population[i] - self.population[j]) + 1e-10
                    acceleration += np.random.rand() * mass[j] * (
                        self.population[j] - self.population[i]
                    ) / (distance ** 3)
            
            # Update velocity and position
            velocity = np.random.rand(self.n_features) + alpha * acceleration
            X_new = self.population[i] + velocity
            
            self.population[i] = self.binary_conversion(X_new)


class BSSA(BinaryMetaheuristic):
    """Binary Salp Swarm Algorithm"""
    
    def update_population(self, iteration):
        c1 = 2 * np.exp(-(4 * iteration / self.max_iterations) ** 2)
        
        for i in range(self.population_size):
            if i == 0:
                # Leader update
                for j in range(self.n_features):
                    c2, c3 = np.random.rand(2)
                    if c3 < 0.5:
                        X_new_j = self.best_solution[j] + c1 * ((
                            1 - 0) * c2 + 0)
                    else:
                        X_new_j = self.best_solution[j] - c1 * ((
                            1 - 0) * c2 + 0)
                    
                    self.population[i, j] = 1 if self.sigmoid(X_new_j) > np.random.rand() else 0
            else:
                # Follower update
                X_new = 0.5 * (self.population[i] + self.population[i - 1])
                self.population[i] = self.binary_conversion(X_new)


class BAO(BinaryMetaheuristic):
    """Binary Aquila Optimizer"""
    
    def update_population(self, iteration):
        t = iteration / self.max_iterations
        
        for i in range(self.population_size):
            if t <= 2/3:
                if np.random.rand() < 0.5:
                    # Expanded exploration
                    X_mean = np.mean(self.population, axis=0)
                    X_new = self.best_solution * (1 - t) + (X_mean - self.best_solution) * np.random.rand()
                else:
                    # Narrowed exploration
                    r = np.random.rand()
                    X_rand = self.population[np.random.randint(0, self.population_size)]
                    X_new = self.best_solution + r * (X_rand - self.population[i])
            else:
                # Exploitation
                alpha, delta = 0.1, 0.1
                QF = t ** ((2 * np.random.rand() - 1))
                G1 = 2 * np.random.rand() - 1
                G2 = 2 * (1 - t)
                
                X_new = (self.best_solution - self.population[i]) * alpha - np.random.rand() + (
                    (1 - 0) * G2) * np.random.rand()
            
            self.population[i] = self.binary_conversion(X_new)


class BAVO(BinaryMetaheuristic):
    """Binary African Vultures Optimization"""
    
    def update_population(self, iteration):
        P = 2 * np.random.rand() - 1
        
        for i in range(self.population_size):
            F = P * (2 * np.random.rand() - 1)
            
            if np.abs(F) >= 1:
                # Exploration
                rand_idx = np.random.randint(0, self.population_size)
                X_new = self.population[rand_idx] - np.abs((
                    2 * np.random.rand()) * self.population[rand_idx] - self.population[i]) * F
            else:
                # Exploitation
                best_idx = np.argmax(self.fitness)
                second_best_idx = np.argsort(self.fitness)[-2]
                
                X_new = (self.population[best_idx] + self.population[second_best_idx]) / 2 - (
                    np.abs((2 * np.random.rand()) * (
                        self.population[best_idx] + self.population[second_best_idx]) / 2 - 
                        self.population[i])) * (2 * np.random.rand() - 1)
            
            self.population[i] = self.binary_conversion(X_new)


class BMOA(BinaryMetaheuristic):
    """Binary Mayfly Optimization Algorithm"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.velocity = np.random.rand(self.population_size, self.n_features)
    
    def update_population(self, iteration):
        a = 2 - iteration * (2 / self.max_iterations)
        
        for i in range(self.population_size):
            # Update velocity
            r1, r2 = np.random.rand(2)
            self.velocity[i] = (a * self.velocity[i] + 
                               r1 * (self.best_solution - self.population[i]) +
                               r2 * (self.population[np.random.randint(0, self.population_size)] - 
                                    self.population[i]))
            
            # Update position
            X_new = self.population[i] + self.velocity[i]
            self.population[i] = self.binary_conversion(X_new)


class BPSO(BinaryMetaheuristic):
    """Binary Particle Swarm Optimization"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.velocity = np.random.rand(self.population_size, self.n_features) * 0.1
        self.pbest = self.population.copy()
        self.pbest_fitness = np.full(self.population_size, -np.inf)
        self.w = 0.9
        self.c1 = 2.0
        self.c2 = 2.0
    
    def update_population(self, iteration):
        # Decrease inertia weight
        self.w = 0.9 - iteration * (0.5 / self.max_iterations)
        
        for i in range(self.population_size):
            # Update personal best
            if self.fitness[i] > self.pbest_fitness[i]:
                self.pbest[i] = self.population[i].copy()
                self.pbest_fitness[i] = self.fitness[i]
            
            # Update velocity
            r1, r2 = np.random.rand(2, self.n_features)
            self.velocity[i] = (self.w * self.velocity[i] +
                               self.c1 * r1 * (self.pbest[i] - self.population[i]) +
                               self.c2 * r2 * (self.best_solution - self.population[i]))
            
            # Update position
            X_new = self.population[i] + self.velocity[i]
            self.population[i] = self.binary_conversion(X_new)


class BGOA(BinaryMetaheuristic):
    """Binary Grasshopper Optimization Algorithm"""
    
    def update_population(self, iteration):
        cmax = 1
        cmin = 0.00001
        c = cmax - iteration * ((cmax - cmin) / self.max_iterations)
        
        for i in range(self.population_size):
            S_i = np.zeros(self.n_features)
            
            for j in range(self.population_size):
                if i != j:
                    distance = np.linalg.norm(self.population[i] - self.population[j]) + 1e-10
                    s = ((1 - 0) / 4) * np.exp(-distance / 0.5) - np.exp(-distance)
                    S_i += c * s * (self.population[j] - self.population[i]) / distance
            
            X_new = c * S_i + self.best_solution
            self.population[i] = self.binary_conversion(X_new)


class BBA(BinaryMetaheuristic):
    """Binary Bat Algorithm"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.velocity = np.zeros((self.population_size, self.n_features))
        self.frequency = np.zeros(self.population_size)
        self.loudness = np.ones(self.population_size)
        self.pulse_rate = np.random.rand(self.population_size)
    
    def update_population(self, iteration):
        alpha = 0.9
        gamma = 0.9
        
        for i in range(self.population_size):
            # Update frequency
            self.frequency[i] = np.random.rand()
            
            # Update velocity and position
            self.velocity[i] = self.velocity[i] + (
                self.population[i] - self.best_solution) * self.frequency[i]
            X_new = self.population[i] + self.velocity[i]
            
            # Local search
            if np.random.rand() > self.pulse_rate[i]:
                epsilon = np.random.rand(self.n_features) * 2 - 1
                X_new = self.best_solution + epsilon * np.mean(self.loudness)
            
            X_new_binary = self.binary_conversion(X_new)
            
            # Accept new solution
            if np.random.rand() < self.loudness[i]:
                self.population[i] = X_new_binary
                self.loudness[i] *= alpha
                self.pulse_rate[i] = self.pulse_rate[i] * (1 - np.exp(-gamma * iteration))


class BWOA(BinaryMetaheuristic):
    """Binary Whale Optimization Algorithm"""
    
    def update_population(self, iteration):
        a = 2 - iteration * (2 / self.max_iterations)
        
        for i in range(self.population_size):
            r = np.random.rand()
            A = 2 * a * r - a
            C = 2 * r
            l = np.random.rand() * 2 - 1
            p = np.random.rand()
            
            if p < 0.5:
                if np.abs(A) < 1:
                    # Encircling prey
                    D = np.abs(C * self.best_solution - self.population[i])
                    X_new = self.best_solution - A * D
                else:
                    # Search for prey
                    rand_idx = np.random.randint(0, self.population_size)
                    D = np.abs(C * self.population[rand_idx] - self.population[i])
                    X_new = self.population[rand_idx] - A * D
            else:
                # Spiral updating position
                D = np.abs(self.best_solution - self.population[i])
                X_new = D * np.exp(l) * np.cos(2 * np.pi * l) + self.best_solution
            
            self.population[i] = self.binary_conversion(X_new)


# Dictionary mapping algorithm names to classes
ALGORITHM_MAP = {
    'BHHO': BHHO,
    'BHGSO': BHGSO,
    'BASO': BASO,
    'BSSA': BSSA,
    'BAO': BAO,
    'BAVO': BAVO,
    'BMOA': BMOA,
    'BPSO': BPSO,
    'BGOA': BGOA,
    'BBA': BBA,
    'BWOA': BWOA
}


def get_algorithm(algorithm_name, n_features, population_size=None, max_iterations=None):
    """
    Factory function to get algorithm instance
    
    Args:
        algorithm_name: Name of algorithm
        n_features: Number of features
        population_size: Population size (optional)
        max_iterations: Max iterations (optional)
        
    Returns:
        Algorithm instance
    """
    if algorithm_name not in ALGORITHM_MAP:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")
    
    return ALGORITHM_MAP[algorithm_name](n_features, population_size, max_iterations)