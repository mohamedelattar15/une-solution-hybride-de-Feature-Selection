import os
import numpy as np
import time
from config import get_config_profile
import sys

# Ensure modules can be imported
sys.path.append('.')
import importlib.util

def import_module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

DataLoader = import_module_from_file("data_loading", "1_data_loading.py").DataLoader
FitnessFunction = import_module_from_file("fitness_function", "2_fitness_function.py").FitnessFunction
GeneticAlgorithm = import_module_from_file("genetic_algorithm", "3_genetic_algorithm.py").GeneticAlgorithm
SimulatedAnnealing = import_module_from_file("simulated_annealing", "4_simulated_annealing.py").SimulatedAnnealing
HybridGASA = import_module_from_file("hybrid_algorithm", "5_hybrid_algorithm.py").HybridGASA

class FeatureSelectionExperiment:
    def __init__(self, output_dir='./results'):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.results = {}
        self.loader = DataLoader()

    def run_dataset(self, name, data, config):
        print(f"\n{'='*40}")
        print(f"Running Experiment on {name.upper()} Dataset")
        print(f"{'='*40}")
        
        X_train, X_test = data['X_train'], data['X_test']
        y_train, y_test = data['y_train'], data['y_test']
        n_features = data['n_features']
        
        fitness = FitnessFunction(X_train, X_test, y_train, y_test)
        ga = GeneticAlgorithm(fitness, n_features, **config['ga'])
        sa = SimulatedAnnealing(fitness, n_features, **config['sa'])
        hybrid = HybridGASA(ga, sa)
        
        start_time = time.time()
        best_sol, best_fit = hybrid.run(n_generations_ga=config['ga']['n_generations'])
        exec_time = time.time() - start_time
        
        accuracy, reduction = fitness.get_accuracy_and_reduction(best_sol)
        n_selected = np.sum(best_sol)
        
        self.results[name] = {
            'best_fitness': best_fit,
            'best_solution': best_sol,
            'accuracy': accuracy,
            'reduction': reduction,
            'n_selected': n_selected,
            'n_total': n_features,
            'time': exec_time,
            'ga_history': ga.history,
            'sa_history': sa.history
        }
        
        # Save detailed text results
        res_file = os.path.join(self.output_dir, f"{name}_detailed_results.txt")
        with open(res_file, 'w') as f:
            f.write(f"Results for {name}\n")
            f.write(f"Fitness: {best_fit:.4f}\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Reduction: {reduction:.2f}%\n")
            f.write(f"Features selected: {n_selected}/{n_features}\n")
            f.write(f"Indices: {np.where(best_sol == 1)[0].tolist()}\n")
            f.write(f"Time: {exec_time:.2f}s\n")
            
        print(f"Done! Saved detailed results to {res_file}")

    def run_all_datasets(self, data_dir='./datasets', profile='medium'):
        datasets = self.loader.load_all_datasets(data_dir)
        config = get_config_profile(profile)
        
        for name, data in datasets.items():
            self.run_dataset(name, data, config)
            
    def generate_report(self):
        import pandas as pd
        data = []
        for name, res in self.results.items():
            data.append({
                'Dataset': name,
                'Fitness': res['best_fitness'],
                'Accuracy': res['accuracy'],
                'Features': f"{res['n_selected']}/{res['n_total']}",
                'Reduction (%)': res['reduction'],
                'Time (s)': res['time']
            })
        df = pd.DataFrame(data)
        csv_path = os.path.join(self.output_dir, 'results_summary.csv')
        df.to_csv(csv_path, index=False)
        print(f"\nSaved summary report to {csv_path}")

if __name__ == '__main__':
    # Make sure datasets folder exists
    if not os.path.exists('./datasets'):
        os.makedirs('./datasets')
        print("Please place wine.data, zoo.data, and krvskp.data in the ./datasets/ directory")
    else:
        exp = FeatureSelectionExperiment()
        # You can pass 'quick' for a fast test run
        exp.run_all_datasets(profile='quick')
        exp.generate_report()
        
        try:
            # We import here using the weird names to make sure python can find them
            import importlib.util
            spec = importlib.util.spec_from_file_location("ResultsAnalyzer", "8_results_analysis.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            analyzer = module.ResultsAnalyzer()
            analyzer.generate_full_report(exp.results, exp.output_dir)
        except Exception as e:
            print(f"Skipping visualization: {e}")
