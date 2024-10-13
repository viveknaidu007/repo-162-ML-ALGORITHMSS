# k-nearest neighbors algorithm

# Importing required modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.colors

# loading dataset
iris = load_iris()

# Iris plants dataset
print(iris.DESCR)
# Iris-Setosa, Iris-Versicolour, Iris-Virginica

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=42)

# Create a KNN Classifier with k=3
knn = KNeighborsClassifier(n_neighbors=3)

# Train the classifier
knn.fit(x_train, y_train)

# Make predictions of the training set
pred = knn.predict(x_test)

print("\n Predictions: ",pred)

# Calculate the accuracy of the classifier
accuracy = knn.score(x_test, y_test)
print("\n Accuracy: ",accuracy)

# Plot petal length against petal width
x = iris.data[:, 2]
y = iris.data[:, 3]
colors = ['red', 'green', 'blue']
plt.scatter(x, y, c=iris.target, cmap= matplotlib.colors.ListedColormap(colors))

# Add labels and title
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.title('Iris plants')
plt.show()