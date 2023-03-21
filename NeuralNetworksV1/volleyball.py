import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def forwardPass(age, weight, height):
    hidden1 = (age * -0.46122 + weight * 0.97314 + height * -0.39203)
    hidden1_after = sigmoid(hidden1 * 0.80109)
    hidden2 = (age * 0.78548 + weight * 2.10584 + height * -0.57847)
    hidden2_after = sigmoid(hidden2 * 0.43529)
    output = (hidden1_after * -0.81546  + hidden2_after * 1.03775) + -0.2368
    return output

print('No1',forwardPass(23,75,176), 'e: 0.798')
print('No2',forwardPass(25,67,180), 'e: 0.800')
print('No3',forwardPass(28,120,175), 'e: -0.0145')
print('No4',forwardPass(22,65,165), 'e: 0.800')
print('No5',forwardPass(46,70,187), 'e: 0.800')


