import numpy as np
import random
import math

class SimulatedAnnealing:
    def __init__(self, fitness_func, n_features, initial_temp=1.0, cooling_rate=0.95, iters_per_temp=200):
        self.fitness_func = fitness_func
        self.n_features = n_features
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.iters_per_temp = iters_per_temp
        self.history = {'temperature': [], 'best_fitness': [], 'acceptance_rate': []}
        
    def get_neighbor(self, solution):
        neighbor = solution.copy()
        # Flip a random bit
        idx = random.randint(0, self.n_features - 1)
        neighbor[idx] = 1 - neighbor[idx]
        
        # Ensure at least one feature
        if np.sum(neighbor) == 0:
            neighbor[random.randint(0, self.n_features - 1)] = 1
        return neighbor
        
    def optimize(self, initial_solution):
        current_solution = initial_solution.copy()
        current_fitness = self.fitness_func.evaluate_solution(current_solution)
        
        best_solution = current_solution.copy()
        best_fitness = current_fitness
        
        temperature = self.initial_temp
        
        while temperature > 0.01:
            accepted = 0
            for _ in range(self.iters_per_temp):
                neighbor = self.get_neighbor(current_solution)
                neighbor_fitness = self.fitness_func.evaluate_solution(neighbor)
                
                # Maximization problem
                delta_E = neighbor_fitness - current_fitness
                
                # Metropolis criteria
                if delta_E > 0 or random.random() < math.exp(delta_E / temperature):
                    current_solution = neighbor
                    current_fitness = neighbor_fitness
                    accepted += 1
                    
                    if current_fitness > best_fitness:
                        best_fitness = current_fitness
                        best_solution = current_solution.copy()
                        
            self.history['temperature'].append(temperature)
            self.history['best_fitness'].append(best_fitness)
            self.history['acceptance_rate'].append(accepted / self.iters_per_temp)
            
            temperature *= self.cooling_rate
            
        return best_solution, best_fitness
