import matplotlib.pyplot as plt
import os
import numpy as np

class ResultsAnalyzer:
    def __init__(self):
        pass

    def plot_convergence(self, name, ga_history, sa_history, output_dir):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # GA Plot
        ax1.plot(ga_history['best_fitness'], label='Best Fitness')
        ax1.plot(ga_history['avg_fitness'], label='Avg Fitness')
        ax1.set_title(f'{name.capitalize()} - GA Phase')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Fitness')
        ax1.legend()
        
        # SA Plot
        ax2.plot(sa_history['best_fitness'], color='green', label='Best Fitness (SA)')
        ax2.set_title(f'{name.capitalize()} - SA Phase')
        ax2.set_xlabel('Temperature Step')
        ax2.set_ylabel('Fitness')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{name}_convergence.png'))
        plt.close()

    def plot_comparison(self, results, output_dir):
        datasets = list(results.keys())
        if not datasets:
            return
            
        accuracies = [res['accuracy'] for res in results.values()]
        reductions = [res['reduction'] / 100.0 for res in results.values()] # scale to 0-1
        
        x = np.arange(len(datasets))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width/2, accuracies, width, label='Accuracy')
        ax.bar(x + width/2, reductions, width, label='Reduction Ratio')
        
        ax.set_ylabel('Scores')
        ax.set_title('Performance Comparison across Datasets')
        ax.set_xticks(x)
        ax.set_xticklabels([d.capitalize() for d in datasets])
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'results_comparison.png'))
        plt.close()

    def generate_full_report(self, results, output_dir='./results'):
        print("Generating visualizations...")
        self.plot_comparison(results, output_dir)
        for name, res in results.items():
            self.plot_convergence(name, res['ga_history'], res['sa_history'], output_dir)
        print(f"Visualizations saved to {output_dir}/")
