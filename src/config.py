import os

GA_CONFIG = {
    'population_size': 30,
    'mutation_rate': 0.1,
    'crossover_rate': 0.8,
    'n_generations': 50,
}

SA_CONFIG = {
    'initial_temperature': 1.0,
    'cooling_rate': 0.95,
    'iterations_per_temperature': 200,
}

FITNESS_CONFIG = {
    'alpha': 0.7, # Weight for accuracy
    'beta': 0.3,  # Weight for feature reduction
    'classifier': 'svm' # Classifier to use
}

def get_config_profile(profile_name='medium'):
    if profile_name == 'quick':
        return {
            'ga': {**GA_CONFIG, 'population_size': 10, 'n_generations': 10},
            'sa': {**SA_CONFIG, 'initial_temperature': 0.5, 'iterations_per_temperature': 50}
        }
    elif profile_name == 'thorough':
        return {
            'ga': {**GA_CONFIG, 'population_size': 50, 'n_generations': 100},
            'sa': {**SA_CONFIG, 'initial_temperature': 2.0, 'iterations_per_temperature': 300}
        }
    return {'ga': GA_CONFIG, 'sa': SA_CONFIG}
