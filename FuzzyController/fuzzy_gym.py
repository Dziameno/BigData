import gym
from simpful import *
import numpy as np

env = gym.make("Pendulum-v1", g=9.81, render_mode='human')
observation = env.reset()

FS = FuzzySystem()

A_1 = FuzzySet(function=Triangular_MF(a=-np.pi, b=-np.pi/3, c=0), term="left")
A_2 = FuzzySet(function=Triangular_MF(a=-np.pi/3, b=0, c=np.pi/3), term="center")
A_3 = FuzzySet(function=Triangular_MF(a=0, b=np.pi/3, c=np.pi), term="right")
FS.add_linguistic_variable("angle", LinguisticVariable([A_1, A_2, A_3], universe_of_discourse=[-np.pi, np.pi]))

AV_1 = FuzzySet(function=Triangular_MF(a=-10, b=-10, c=0), term="negative")
AV_2 = FuzzySet(function=Triangular_MF(a=-10, b=0, c=10), term="zero")
AV_3 = FuzzySet(function=Triangular_MF(a=0, b=10, c=10), term="positive")
FS.add_linguistic_variable("angular_velocity", LinguisticVariable([AV_1, AV_2, AV_3], universe_of_discourse=[-10, 10]))

T_1 = FuzzySet(function=Triangular_MF(a=-50, b=-50, c=0), term="negative")
T_2 = FuzzySet(function=Triangular_MF(a=-50, b=0, c=50), term="zero")
T_3 = FuzzySet(function=Triangular_MF(a=0, b=50, c=50), term="positive")
FS.add_linguistic_variable("torque", LinguisticVariable([T_1, T_2, T_3], universe_of_discourse=[-50, 50]))

R1 = "IF (angle IS left) AND (angular_velocity IS negative) THEN (torque IS positive)"
R2 = "IF (angle IS left) AND (angular_velocity IS zero) THEN (torque IS positive)"
R3 = "IF (angle IS left) AND (angular_velocity IS positive) THEN (torque IS positive)"

R4 = "IF (angle IS center) AND (angular_velocity IS negative) THEN (torque IS positive)"
R5 = "IF (angle IS center) AND (angular_velocity IS zero) THEN (torque IS zero)"
R6 = "IF (angle IS center) AND (angular_velocity IS positive) THEN (torque IS negative)"

R7 = "IF (angle IS right) AND (angular_velocity IS negative) THEN (torque IS negative)"
R8 = "IF (angle IS right) AND (angular_velocity IS zero) THEN (torque IS negative)"
R9 = "IF (angle IS right) AND (angular_velocity IS positive) THEN (torque IS negative)"

FS.add_rules([R1, R2, R3, R4, R5, R6, R7, R8, R9])

# FS.plot_variable("angle")
# FS.plot_variable("angular_velocity")
# FS.plot_variable("torque")

env.reset()
for i in range(1000):
    env.render()
    angle = env.state[0]
    angular_velocity = env.state[1]
    FS.set_variable("angle", angle, True)
    FS.set_variable("angular_velocity", angular_velocity, True)
    torque = FS.inference()
    torque = torque['torque']
    env.step([torque])
env.close()