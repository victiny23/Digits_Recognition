import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# get the MNIST digits dataset
mnist = fetch_openml('mnist_784')

# show example images
plt.figure(figsize=(25, 20))
for index, (image, label) in enumerate(zip(mnist.data[:25], mnist.target[:25])):
    plt.subplot(5, 5, index+1)
    plt.imshow(np.reshape(image, (28, 28)), cmap='gray')
    plt.title('Number: %s' % label)
    plt.xticks([])
    plt.yticks([])
plt.suptitle('Example Images', fontsize=25)
"""
plt.figure(figsize=(20, 4))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(np.reshape(mnist.data[i], (28,28)), cmap='gray')
    plt.title('Number: {}'.format(mnist.target[i]))
"""

# divide data into training set and test set
# x: image, y: label
x_train, x_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=0.2)
# x_train array of (56000, 784), image of 28 x 28
# x_test array of (14000, 784), image of 28 x 28

# show example images from training set and test set
# create 2 big, invisible subplots, each with a title
fig, big_axes = plt.subplots(figsize=(20, 10),
                            nrows=2, ncols=1, sharey=True)
suptitle = ['Training Set Example Images',
            'Test Set Example Images']
for row, big_ax in enumerate(big_axes, start=1):
    big_ax.set_title(suptitle[row-1], fontsize=20)
    big_ax.tick_params(labelcolor=(1., 1., 1., 0.0), top='off',
                       bottom='off', left='off', right='off')
    big_ax._frameon = False
# add plots of example images
for i in range(5):
    ax1 = fig.add_subplot(2, 5, i+1)
    ax1.imshow(np.reshape(x_train[i], (28, 28)), cmap='gray')
    ax1.set_title('Number: {}'.format(y_train[i]))
    ax2 = fig.add_subplot(2, 5, i+6)
    ax2.imshow(np.reshape(x_test[i], (28, 28)), cmap='gray')
    ax2.set_title('Number: {}'.format(y_test[i]))
plt.show()

# LinearRegression Model
mdl = LogisticRegression(solver='lbfgs')
mdl.fit(x_train, y_train)
predictions = mdl.predict(x_test)
score = mdl.score(x_test, y_test)
# print(score)
# score is an np.array of 1 element
print('{:.2}'.format(score.item(0)))
# another way to calculate the score
print('{:.3}'.format(sum(predictions == y_test) / len(y_test)))

# the value of prediction for a given image
# mdl.predict([x_test[0]])[0]
# ['6'] # np.array
# mdl.predict([x_test[0]])[0]
# 6  # str

# get the confusion matrix
cm = metrics.confusion_matrix(y_test, predictions)
# cm is an np.array of 10 x 10
# in row 0, how many times 0 (the truth) is recognized as
# 0, 1, ..., 9
# the sum of each row is the count for each number

plt.figure(figsize=(10, 10))
plt.imshow(cm, cmap='Pastel1')
plt.title('Confusion Matrix of MNIST Data', fontsize=20)
plt.xticks(np.arange(10))
plt.yticks(np.arange(10))
plt.xlabel('Prediction', fontsize=15)
plt.ylabel('Truth', fontsize=15)
plt.colorbar()
width, height = cm.shape
for x in range(width):
    for y in range(height):
        plt.annotate(cm[x][y], xy=(y, x), ha='center', va='center')
plt.show()

# convert confusion_matrix into percentage
cm_pct = cm.astype('float') /(cm.sum(axis=1)) * 100

# prediction accuracy
plt.figure(figsize=(10, 10))
plt.imshow(cm_pct, cmap='RdBu')
plt.title('Accuracy of Digits Recognition', fontsize=20)
plt.xticks(np.arange(10))
plt.yticks(np.arange(10))
plt.xlabel('Prediction', fontsize=15)
plt.ylabel('Truth', fontsize=15)
cbar = plt.colorbar()
cbar.ax.set_title('Accuracy (%)')
width, height = cm.shape
for x in range(width):
    for y in range(height):
        plt.annotate('{:.2f}'.format(cm_pct.item(x, y)),
                     xy=(y, x), ha='center', va='center',
                     color='w')
