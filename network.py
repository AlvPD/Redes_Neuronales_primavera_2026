# %load network.py

"""
network.py
~~~~~~~~~~
FUNCIONA

Un módulo para implementar el algoritmo de aprendizaje de descenso de gradiente
estocástico para una red neuronal prealimentada (feedforward). Los gradientes se
calculan usando retropropagación (backpropagation). He priorizado que el código 
sea simple, legible y fácil de modificar. No está optimizado y omite muchas 
características deseables.
"""

#### Bibliotecas
# Biblioteca estándar
import random

# Bibliotecas de terceros
import numpy as np

class Network(object):

    def __init__(self, sizes):
        """La lista ``sizes`` contiene el número de neuronas en las capas
        respectivas de la red. Por ejemplo, [2, 3, 1] sería una red de tres capas.
        Los sesgos (biases) y pesos se inicializan aleatoriamente usando una
        distribución gaussiana ($\mu=0, \sigma=1$). La primera capa se asume de entrada
        y no lleva sesgos."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Devuelve la salida de la red si ``a`` es la entrada."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Entrena la red usando descenso de gradiente estocástico por mini-lotes.
        ``training_data`` es una lista de tuplas ``(x, y)``. Si se proporciona
        ``test_data``, la red se evaluará tras cada época para rastrear el progreso."""

        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {} : {} / {}".format(j,self.evaluate(test_data),n_test))
            else:
                print("Epoch {} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """Actualiza los pesos y sesgos mediante retropropagación en un solo mini-lote.
        ``eta`` es la tasa de aprendizaje."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Devuelve una tupla ``(nabla_b, nabla_w)`` que representa el gradiente
        de la función de costo. Son listas de arreglos de numpy por capa,
        similar a self.bias y self.weights"""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # Paso hacia adelante (feedforward)
        activation = x
        activations = [x] # ista para almacenar todas las activaciones por capa
        zs = [] # Lista para almacenar todos los vectores z por capa
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # Paso hacia atrás (backward pass)
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # l = 1 es la última capa, l = 2 la penúltima, y así sucesivamente
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Devuelve el número de entradas de prueba para las cuales la red 
        obtiene el resultado correcto (el índice de la neurona con mayor activación)."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Devuelve el vector de derivadas parciales $\partial C_x / \partial a$."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """La función sigmoide."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivada de la función sigmoide."""
    return sigmoid(z)*(1-sigmoid(z))