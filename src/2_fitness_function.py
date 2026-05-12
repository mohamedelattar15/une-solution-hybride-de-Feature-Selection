import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

class FitnessFunction:
    def __init__(self, X_train, X_test, y_train, y_test, alpha=0.7, beta=0.3, classifier='svm'):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.alpha = alpha
        self.beta = beta
        self.n_total = X_train.shape[1]
        self.evaluations = 0
        
        if classifier == 'svm':
            self.clf = SVC(kernel='linear', random_state=42)
        elif classifier == 'dt':
            self.clf = DecisionTreeClassifier(random_state=42)
        else:
            self.clf = GaussianNB()

    def evaluate_solution(self, chromosome):
        self.evaluations += 1
        selected_features = np.where(chromosome == 1)[0]
        n_selected = len(selected_features)
        
        if n_selected == 0:
            return 0.0 # Invalid solution
            
        X_train_sel = self.X_train[:, selected_features]
        X_test_sel = self.X_test[:, selected_features]
        
        self.clf.fit(X_train_sel, self.y_train)
        preds = self.clf.predict(X_test_sel)
        accuracy = accuracy_score(self.y_test, preds)
        
        # Fitness = alpha * Accuracy - beta * (n_selected / n_total)
        # We maximize this, meaning high accuracy and low number of features
        fitness = self.alpha * accuracy - self.beta * (n_selected / self.n_total)
        return max(fitness, 0.0) # Ensure positive fitness

    def get_accuracy_and_reduction(self, chromosome):
        selected_features = np.where(chromosome == 1)[0]
        n_selected = len(selected_features)
        
        if n_selected == 0:
            return 0.0, 0.0
            
        X_train_sel = self.X_train[:, selected_features]
        X_test_sel = self.X_test[:, selected_features]
        
        self.clf.fit(X_train_sel, self.y_train)
        preds = self.clf.predict(X_test_sel)
        accuracy = accuracy_score(self.y_test, preds)
        
        reduction = 100.0 * (1.0 - (n_selected / self.n_total))
        return accuracy, reduction
        
    def reset_evaluation_count(self):
        self.evaluations = 0
