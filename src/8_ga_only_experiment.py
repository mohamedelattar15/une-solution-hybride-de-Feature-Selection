import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Ensure modules can be imported
import sys
import importlib.util
sys.path.append('.')

def import_module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

DataLoader = import_module_from_file("data_loading", "1_data_loading.py").DataLoader
FitnessFunction = import_module_from_file("fitness_function", "2_fitness_function.py").FitnessFunction
GeneticAlgorithm = import_module_from_file("genetic_algorithm", "3_genetic_algorithm.py").GeneticAlgorithm

def run_exercise_1():
    print("="*50)
    print("🚀 EXERCICE 1 : Feature Selection avec AG Binaire")
    print("Dataset : Breast Cancer (breast_dataset.csv)")
    print("="*50)

    # 1. Chargement des données
    loader = DataLoader()
    # Note: On utilise le chemin relatif correct selon l'emplacement d'exécution
    data_dir = '../datasets'
    if not os.path.exists(data_dir):
        data_dir = './datasets' # Fallback
        
    datasets = loader.load_all_datasets(data_dir)
    
    if 'breast' not in datasets:
        print(f"❌ Erreur : Le dataset Breast Cancer est introuvable dans {data_dir}")
        return

    data = datasets['breast']
    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train'], data['y_test']
    n_features = data['n_features']
    
    print(f"✅ Données chargées : {X_train.shape[0]} échantillons, {n_features} caractéristiques.")

    # 2. Initialisation de la fonction de fitness
    fitness_eval = FitnessFunction(X_train, X_test, y_train, y_test, alpha=0.7)

    # 3. Configuration et exécution de l'Algorithme Génétique
    print("\n🧬 Démarrage de l'Algorithme Génétique...")
    ga = GeneticAlgorithm(
        fitness_func=fitness_eval,
        n_features=n_features,
        population_size=50,
        mutation_rate=0.01,
        crossover_rate=0.8,
        n_generations=100
    )

    start_time = datetime.now()
    best_sol, best_fit = ga.evolve(n_generations=100)
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # 4. Évaluation finale
    final_acc, final_red = fitness_eval.get_accuracy_and_reduction(best_sol)
    n_selected = int(np.sum(best_sol))

    print("\n" + "="*30)
    print("📊 RÉSULTATS FINAUX (EXERCICE 1)")
    print("="*30)
    print(f"Meilleure Fitness      : {best_fit:.4f}")
    print(f"Précision (Accuracy)   : {final_acc*100:.2f}%")
    print(f"Features sélectionnées : {n_selected} / {n_features}")
    print(f"Taux de réduction      : {final_red:.2f}%")
    print(f"Temps d'exécution      : {duration:.2f} secondes")
    print("="*30)

    # 5. Visualisation de la convergence
    plt.figure(figsize=(10, 6))
    plt.plot(ga.history['best_fitness'], label='Meilleure Fitness', color='blue', linewidth=2)
    plt.plot(ga.history['avg_fitness'], label='Fitness Moyenne', color='green', linestyle='--', alpha=0.7)
    plt.title('Exercice 1 : Convergence de l\'Algorithme Génétique (Breast Cancer)')
    plt.xlabel('Générations')
    plt.ylabel('Score de Fitness')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Sauvegarde dans le sous-dossier dédié : results/ex1_ga/
    base_dir = '../results' if os.path.exists('../results') else './results'
    res_dir = os.path.join(base_dir, 'ex1_ga')
    os.makedirs(res_dir, exist_ok=True)
    
    plt.savefig(os.path.join(res_dir, 'ga_convergence.png'))
    print(f"\n📈 Graphique de convergence sauvegardé dans {res_dir}/ga_convergence.png")
    
    # Sauvegarde des résultats dans un fichier texte
    with open(os.path.join(res_dir, 'ex1_results.txt'), 'w') as f:
        f.write("EXERCICE 1 : RÉSULTATS GA SEUL (BREAST CANCER)\n")
        f.write("="*40 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Précision: {final_acc*100:.2f}%\n")
        f.write(f"Réduction: {final_red:.2f}%\n")
        f.write(f"Nombre de features: {n_selected}/{n_features}\n")
        f.write(f"Temps: {duration:.2f}s\n")

if __name__ == "__main__":
    run_exercise_1()
