import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class GeneticAlgorithm:
    def __init__(self, population_size=50, mutation_rate=0.1, 
                 crossover_rate=0.8, num_generations=100):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.num_generations = num_generations
    
    def initialize_population(self, num_features):
        """Create initial random population"""
        return np.random.randint(2, size=(self.population_size, num_features))
    
    def fitness_function(self, chromosome, X_train, y_train, X_val, y_val):
        """Evaluate fitness using selected features"""
        selected_features = np.where(chromosome == 1)[0]
        
        if len(selected_features) == 0:
            return 0.0
        
        X_train_selected = X_train[:, selected_features]
        X_val_selected = X_val[:, selected_features]
        
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X_train_selected, y_train)
        
        y_pred = clf.predict(X_val_selected)
        accuracy = accuracy_score(y_val, y_pred)
        
        return accuracy
    
    def selection(self, population, fitness_scores):
        """Select parents using tournament selection"""
        parents = []
        for _ in range(2):
            tournament = np.random.choice(len(population), size=5)
            winner = tournament[np.argmax(fitness_scores[tournament])]
            parents.append(population[winner])
        return parents
    
    def crossover(self, parent1, parent2):
        """Single-point crossover"""
        if np.random.rand() < self.crossover_rate:
            point = np.random.randint(1, len(parent1))
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])
            return child1, child2
        return parent1.copy(), parent2.copy()
    
    def mutation(self, chromosome):
        """Bit-flip mutation"""
        for i in range(len(chromosome)):
            if np.random.rand() < self.mutation_rate:
                chromosome[i] = 1 - chromosome[i]
        return chromosome
    
    def evolve(self, X, y):
        """Run genetic algorithm for feature selection"""
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        num_features = X.shape[1]
        population = self.initialize_population(num_features)
        
        best_chromosome = None
        best_fitness = 0.0
        
        for generation in range(self.num_generations):
            # Evaluate fitness
            fitness_scores = np.array([
                self.fitness_function(chrom, X_train, y_train, X_val, y_val)
                for chrom in population
            ])
            
            # Track best solution
            max_fitness = np.max(fitness_scores)
            if max_fitness > best_fitness:
                best_fitness = max_fitness
                best_chromosome = population[np.argmax(fitness_scores)].copy()
            
            # Create new generation
            new_population = []
            while len(new_population) < self.population_size:
                parents = self.selection(population, fitness_scores)
                child1, child2 = self.crossover(parents[0], parents[1])
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)
                new_population.extend([child1, child2])
            
            population = np.array(new_population[:self.population_size])
            
            if (generation + 1) % 10 == 0:
                print(f"Generation {generation+1}: Best Fitness = {best_fitness:.4f}")
        
        return best_chromosome, best_fitness
