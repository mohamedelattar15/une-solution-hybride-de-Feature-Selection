class HybridGASA:
    def __init__(self, ga, sa):
        self.ga = ga
        self.sa = sa
        
    def run(self, n_generations_ga=50):
        print("Starting Genetic Algorithm Phase (Global Search)...")
        best_ga_sol, best_ga_fit = self.ga.evolve(n_generations=n_generations_ga)
        print(f"Best fitness after GA: {best_ga_fit:.4f}")
        
        print("Starting Simulated Annealing Phase (Local Refinement)...")
        best_hybrid_sol, best_hybrid_fit = self.sa.optimize(best_ga_sol)
        print(f"Best fitness after SA: {best_hybrid_fit:.4f}")
        
        return best_hybrid_sol, best_hybrid_fit
