import sys
import importlib.util
sys.path.append('.')

import numpy as np
from config import get_config_profile

# Import modules with numbers in their names
def import_module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

mod1 = import_module_from_file("data_loading", "1_data_loading.py")
mod2 = import_module_from_file("fitness_function", "2_fitness_function.py")
mod3 = import_module_from_file("genetic_algorithm", "3_genetic_algorithm.py")
mod4 = import_module_from_file("simulated_annealing", "4_simulated_annealing.py")
mod5 = import_module_from_file("hybrid_algorithm", "5_hybrid_algorithm.py")

def test_pipeline():
    print("Initializing test...")
    # Generate dummy dataset instead of loading to guarantee the test runs
    X_train = np.random.rand(100, 10)
    X_test = np.random.rand(30, 10)
    y_train = np.random.randint(0, 2, 100)
    y_test = np.random.randint(0, 2, 30)
    n_features = 10
    
    print("Testing FitnessFunction...")
    fitness = mod2.FitnessFunction(X_train, X_test, y_train, y_test)
    
    print("Testing Genetic Algorithm...")
    ga = mod3.GeneticAlgorithm(fitness, n_features, population_size=10, n_generations=5)
    
    print("Testing Simulated Annealing...")
    sa = mod4.SimulatedAnnealing(fitness, n_features, initial_temp=0.5, iters_per_temp=10)
    
    print("Testing Hybrid Algorithm...")
    hybrid = mod5.HybridGASA(ga, sa)
    best_sol, best_fit = hybrid.run(n_generations_ga=5)
    
    print(f"\nTEST SUCCESSFUL! Best Fitness: {best_fit:.4f}")
    print(f"Selected Features: {np.where(best_sol == 1)[0]}")

if __name__ == '__main__':
    test_pipeline()
