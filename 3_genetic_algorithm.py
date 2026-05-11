import numpy as np
import random

class GeneticAlgorithm:
    def __init__(self, fitness_func, n_features, population_size=30, mutation_rate=0.1, crossover_rate=0.8):
        self.fitness_func = fitness_func
        self.n_features = n_features
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.history = {'best_fitness': [], 'avg_fitness': []}
        
    def init_population(self):
        # Initialize randomly with 1s and 0s
        return np.random.randint(2, size=(self.population_size, self.n_features))
        
    def tournament_selection(self, population, fitnesses, k=3):
        selected_indices = np.random.choice(len(population), k, replace=False)
        best_index = selected_indices[np.argmax(fitnesses[selected_indices])]
        return population[best_index].copy()
        
    def crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate:
            point = random.randint(1, self.n_features - 1)
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])
            return child1, child2
        return parent1.copy(), parent2.copy()
        
    def mutate(self, chromosome):
        for i in range(self.n_features):
            if random.random() < self.mutation_rate:
                chromosome[i] = 1 - chromosome[i] # Bit-flip
        # Ensure at least one feature is selected
        if np.sum(chromosome) == 0:
            chromosome[random.randint(0, self.n_features-1)] = 1
        return chromosome
        
    def evolve(self, n_generations=50):
        population = self.init_population()
        best_solution = None
        best_fitness = -1
        
        for gen in range(n_generations):
            fitnesses = np.array([self.fitness_func.evaluate_solution(ind) for ind in population])
            
            # Record keeping
            current_best_idx = np.argmax(fitnesses)
            if fitnesses[current_best_idx] > best_fitness:
                best_fitness = fitnesses[current_best_idx]
                best_solution = population[current_best_idx].copy()
                
            self.history['best_fitness'].append(best_fitness)
            self.history['avg_fitness'].append(np.mean(fitnesses))
            
            new_population = []
            # Elitism: keep best
            new_population.append(best_solution.copy())
            
            while len(new_population) < self.population_size:
                p1 = self.tournament_selection(population, fitnesses)
                p2 = self.tournament_selection(population, fitnesses)
                c1, c2 = self.crossover(p1, p2)
                new_population.extend([self.mutate(c1), self.mutate(c2)])
                
            population = np.array(new_population[:self.population_size])
            
        return best_solution, best_fitness
