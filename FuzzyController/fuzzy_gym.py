import gym
from simpful import *
import numpy as np

env = gym.make("Pendulum-v1", g=9.81, render_mode='human')
observation = env.reset()

FS = FuzzySystem()

angle = AutoTriangle(3, terms=['left', 'center', 'right'], universe_of_discourse=[-np.pi, np.pi])
FS.add_linguistic_variable("angle", angle)

angular_velocity = AutoTriangle(3, terms=['negative', 'zero', 'positive'], universe_of_discourse=[-8, 8])
FS.add_linguistic_variable("angular_velocity", angular_velocity)

torque = AutoTriangle(3, terms=['negative', 'zero', 'positive'], universe_of_discourse=[-2, 2])
FS.add_linguistic_variable("torque", torque)

FS.add_rules([
    "IF (angle IS left) AND (angular_velocity IS negative) THEN (torque IS positive)",
    "IF (angle IS left) AND (angular_velocity IS zero) THEN (torque IS positive)",
    "IF (angle IS left) AND (angular_velocity IS positive) THEN (torque IS positive)",
    "IF (angle IS center) AND (angular_velocity IS negative) THEN (torque IS positive)",
    "IF (angle IS center) AND (angular_velocity IS zero) THEN (torque IS zero)",
    "IF (angle IS center) AND (angular_velocity IS positive) THEN (torque IS negative)",
    "IF (angle IS right) AND (angular_velocity IS negative) THEN (torque IS negative)",
    "IF (angle IS right) AND (angular_velocity IS zero) THEN (torque IS negative)",
    "IF (angle IS right) AND (angular_velocity IS positive) THEN (torque IS negative)"
    ])

# FS.plot_variable("angle")
# FS.plot_variable("angular_velocity")
# FS.plot_variable("torque")


for i in range(100):
    env.reset()
    env.render()

    angle = env.state[0]
    angular_velocity = env.state[1]
    FS.set_variable("angle", angle, True)
    FS.set_variable("angular_velocity", angular_velocity, True)
    torque = FS.inference()

env.close()