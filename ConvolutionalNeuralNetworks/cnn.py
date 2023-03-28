import numpy as np
from matplotlib import pyplot as plt

# tworzymy tablice o wymiarach 128x128x3 (3 kanaly to RGB)
# uzupelnioną zerami = kolor czarny
data = np.zeros((128, 128, 3), dtype=np.uint8)


# chcemy zeby obrazek byl czarnobialy,
# wiec wszystkie trzy kanaly rgb uzupelniamy tymi samymi liczbami
# napiszmy do tego funkcje
def draw(img, x, y, color):
    img[x, y] = [color, color, color]


# zamalowanie 4 pikseli w lewym górnym rogu
draw(data, 5, 5, 100)
draw(data, 6, 6, 100)
draw(data, 5, 6, 255)
draw(data, 6, 5, 255)


# rysowanie kilku figur na obrazku
for i in range(128):
    for j in range(128):
        if (i-64)**2 + (j-64)**2 < 900:
            draw(data, i, j, 200)
        elif i > 100 and j > 100:
            draw(data, i, j, 255)
        elif (i-15)**2 + (j-110)**2 < 25:
            draw(data, i, j, 150)
        elif (i-15)**2 + (j-110)**2 == 25 or (i-15)**2 + (j-110)**2 == 26:
            draw(data, i, j, 255)

# konwersja macierzy na obrazek i wyświetlenie
plt.imshow(data, interpolation='nearest')
plt.savefig('cnn.png')
#plt.show()

#vertical filter
v_filtered_img = np.zeros((128, 128, 3))
v_filter = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])

for i in range(1, 127):
    for j in range(1, 127):
        for k in range(3):
                v_filtered_img[i, j] = np.sum(data[i-1:i+2, j-1:j+2, k] * v_filter)

plt.imshow(v_filtered_img, interpolation='nearest')
plt.savefig('cnn_vertical_filtered.png')

#vertical linear ratification (ReLU)
def relu(x):
    if x < 0:
        v_filtered_img[i, j] = [255,255,255]
    if x >= 0:
        return x

for i in range(1, 127):
    for j in range(1, 127):
        for k in range(3):
            relu(v_filtered_img[i, j, k])

plt.imshow(v_filtered_img, interpolation='nearest')
plt.savefig('cnn_relu_vertical.png')

#horizontal filter
h_filtered_img = np.zeros((128, 128, 3))
h_filter = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

for i in range(1, 127):
    for j in range(1, 127):
        for k in range(3):
            h_filtered_img[i, j, k] = np.sum(data[i-1:i+2, j-1:j+2, k] * h_filter)

plt.imshow(h_filtered_img, interpolation='nearest')
plt.savefig('cnn_horizontal_filtered.png')

#horizontal (ReLU)
def relu(x):
    if x < 0:
        h_filtered_img[i, j] = [255,255,255]
    if x >= 0:
        return x

for i in range(1, 127):
    for j in range(1, 127):
        for k in range(3):
            relu(h_filtered_img[i, j, k])

plt.imshow(h_filtered_img, interpolation='nearest')
plt.savefig('cnn_relu_horizontal.png')

#crosswise filter with Sobel operator
c_filtered_img = np.zeros((128, 128, 3))
c_filter = np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]])

for i in range(1, 127):
    for j in range(1, 127):
        for k in range(3):
            c_filtered_img[i, j, k] = np.sum(data[i-1:i+2, j-1:j+2, k] * c_filter)

plt.imshow(c_filtered_img, interpolation='nearest')
plt.savefig('cnn_crosswise_filtered.png')

#crosswise (ReLU)
def relu(x):
    if x < 0:
        c_filtered_img[i, j] = [255,255,255]
    if x >= 0:
        return x

for i in range(1, 127):
    for j in range(1, 127):
        for k in range(3):
            relu(c_filtered_img[i, j, k])

plt.imshow(c_filtered_img, interpolation='nearest')
plt.savefig('cnn_relu_crosswise.png')