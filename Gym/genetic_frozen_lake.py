import gymnasium as gym
import pygame
import numpy as np
from gym.envs.toy_text.frozen_lake import generate_random_map
import pygame.locals as pylocals
import os.path

population_size = 500
generations = 500
elite_size = 5
mutation_rate = 0.05

# random_size = np.random.randint(5, 7)
random_size = 5
# random_floes = np.random.random()
random_floes = 0.1
random_map = generate_random_map(size=random_size, p=random_floes)

env = gym.make("FrozenLake-v1", desc=random_map, is_slippery=False)
# env = gym.make("FrozenLake-v1", desc=random_map, is_slippery=False, render_mode="human")


def evaluate_fitness(individual):
    fitness = 0
    env.reset()
    for action in individual:
        location, reward, isDead, info, terminated = env.step(action)

        if isDead:
            fitness -= 1
            env.reset()
        elif reward == 1:
            fitness += 15
            break
        else:
            fitness += 0

    return fitness


def initialize_population():
    population = []
    for _ in range(population_size):
        individual = np.random.randint(4, size=100)
        population.append(individual)
    return population


def crossover(parent1, parent2):
    crossover_point = np.random.randint(len(parent1))
    child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    return child


def mutate(individual):
    for i in range(len(individual)):
        if np.random.random() < mutation_rate:
            individual[i] = np.random.randint(4)
    return individual


def evolve(population):
    parents = population[:elite_size]
    for _ in range(population_size - elite_size):
        parent1 = population[np.random.randint(len(population[:elite_size]))]
        parent2 = population[np.random.randint(len(population))]
        child = crossover(parent1, parent2)
        child = mutate(child)
        parents.append(child)
    return parents


def save_best_individual(individual):
    np.save("best_individual.npy", individual)


def load_best_individual():
    return np.load("best_individual.npy")


def main():
    pygame.init()

    if os.path.isfile("best_individual.npy"):
        population = [load_best_individual()]
    else:
        population = initialize_population()

    best_fitness_overall = -float('inf')

    for generation in range(generations):
        fitness_scores = []
        for individual in population:
            fitness_scores.append(evaluate_fitness(individual))

        best_fitness = max(fitness_scores)
        if best_fitness > best_fitness_overall:
            best_fitness_overall = best_fitness

        best_individual = population[np.argmax(fitness_scores)]
        population = evolve(population)
        print("Generation:", generation + 1, "Highest fitness:", best_fitness)

    print("Best fitness overall:", best_fitness_overall)
    save_best_individual(best_individual)

if __name__ == "__main__":
    main()