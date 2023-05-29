import gymnasium as gym
import pygame
import numpy as np
from gym.envs.toy_text.frozen_lake import generate_random_map
import os.path

population_size = 2000
generations = 500
elite_size = 50
mutation_rate = 0.2

MAP = { '8x8': ["SFFFFFFF"],}

# random_size = np.random.randint(5, 7)
# random_floes = np.random.random()
#random_map = generate_random_map(size=random_size, p=random_floes)

env = gym.make("FrozenLake-v1", map_name='8x8', is_slippery=False)
#env = gym.make("FrozenLake-v1", map_name='8x8', is_slippery=False, render_mode="human")


def evaluate_fitness(individual):
    fitness = 0
    success_count = 0
    failure_count = 0
    env.reset()
    for action in individual:
        location, reward, isDead, info, terminated = env.step(action)

        if reward == 1:
            fitness += 25
            success_count += 1
            env.reset()
        if isDead == True:
            failure_count += 1
            env.reset()
            break
        else:
            fitness -= 1

    return fitness, success_count, failure_count


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


def save_best_individual(individual):
    np.save("best_individual.npy", individual)


def main():
    pygame.init()

    if os.path.isfile("best_individual.npy"):
        population = [load_best_individual()]
    else:
        population = initialize_population()

    best_fitness_overall = -float('inf')
    best_individual_overall = None
    best_generation_overall = -1
    total_success_count = 0
    total_failure_count = 0

    for generation in range(generations):
        fitness_scores = []
        generation_success_count = 0
        generation_failure_count = 0
        for individual in population:
            fitness, success_count, failure_count = evaluate_fitness(individual)
            fitness_scores.append(fitness)
            generation_success_count += success_count
            generation_failure_count += failure_count

        current_best_fitness = max(fitness_scores)
        if current_best_fitness > best_fitness_overall:
            best_fitness_overall = current_best_fitness
            best_individual_overall = population[np.argmax(fitness_scores)]
            best_generation_overall = generation + 1

        population = evolve(population)
        print("Gen:", generation + 1, "Highest fitness:", current_best_fitness,
              "Acc: {:.2f}%".format((generation_success_count / (generation_success_count + generation_failure_count)) * 100))

        total_success_count += generation_success_count
        total_failure_count += generation_failure_count

    print("Best fitness overall:", best_fitness_overall)
    print("Best generation overall:", best_generation_overall)
    print("Accuracy of all generations: {:.2f}%".format((total_success_count / (total_success_count + total_failure_count))*100))
    save_best_individual(best_individual_overall)


if __name__ == "__main__":
    main()