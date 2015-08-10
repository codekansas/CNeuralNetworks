#include "matrix.h"
#include <math.h>
#include <vector>

const double e = 2.7182818284;

#define f(x) (1.0 / (1.0 + pow(e, -x)))
#define df(x) (f(x) * (1 - f(x)))

class BackPropagation {
	private:
		int N_LAYERS, NEURONS_PER_LAYER, N_INPUTS, N_OUTPUTS;
		double MIN_WEIGHT, MAX_WEIGHT;
		vector<Matrix> weights, biases;
	public:
		double LEARNING_RATE;
		BackPropagation(int, int, int, int, double);
		Matrix eval(Matrix);
		void train(Matrix, Matrix);
};

Matrix activationFunction(Matrix m) {
	Matrix n = *new Matrix(m.width, m.height);
	for (int i = 0; i < m.width; i++) {
		for (int j = 0; j < m.height; j++) {
			n.set(i, j, f(m.get(i, j)));
		}
	}
	return n;
}

Matrix BackPropagation::eval(Matrix m) {
	for (int i = 0; i < N_LAYERS; i++) {
		m = activationFunction(m.times(weights[i]).plus(biases[i]));
	}
	return m;
}

void BackPropagation::train(Matrix input, Matrix output) {
	vector<Matrix> outputs(N_LAYERS);
	vector<Matrix> deltas(N_LAYERS);
	
	// Feed forward
	outputs[0] = activationFunction(input.times(weights[0]));
	for (int i = 1; i < N_LAYERS; i++) {
		outputs[i] = activationFunction(outputs[i-1].times(weights[i]).plus(biases[i]));
	}
	
	// Feed backward
	deltas[N_LAYERS-1] = output.T().minus(outputs[N_LAYERS-1]).dot(outputs[N_LAYERS-1].times(-1).plus(1).dot(outputs[N_LAYERS-1]));
	for (int i = N_LAYERS - 2; i >= 0; i--) {
		deltas[i] = deltas[i+1].times(weights[i+1].T()).dot(outputs[i].times(-1).plus(1).dot(outputs[i]));
	}
	
	// Update weights
	weights[0] = weights[0].plus(input.T().times(deltas[0]).times(LEARNING_RATE));
	for (int i = 1; i < N_LAYERS; i++) {
		weights[i] = weights[i].plus(outputs[i-1].T().times(deltas[i]).times(LEARNING_RATE));
	}
	
	// Update biases
	for (int i = 0; i < N_LAYERS; i++) {
		biases[i] = biases[i].plus(deltas[i].times(LEARNING_RATE));
	}
}

BackPropagation::BackPropagation(int layers, int neurons, int inputs, int outputs, double lr) {
	layers++;
	
	N_LAYERS = layers;
	NEURONS_PER_LAYER = neurons;
	N_INPUTS = inputs;
	N_OUTPUTS = outputs;
	LEARNING_RATE = lr;
	
	MIN_WEIGHT = -1;
	MAX_WEIGHT = 1;
	
	vector<Matrix> w(N_LAYERS);
	for (int i = 0; i < N_LAYERS; i++) {
		if (i == 0) {
			w[i] = *new Matrix(neurons, inputs);
		} else if (i == N_LAYERS - 1) {
			w[i] = *new Matrix(outputs, neurons);
		} else {
			w[i] = *new Matrix(neurons, neurons);
		}
		randomDoubles(&w[i], MIN_WEIGHT, MAX_WEIGHT);
	}
	weights = w;
	
	vector<Matrix> b(N_LAYERS);
	for (int i = 0; i < N_LAYERS; i++) {
		if (i == N_LAYERS - 1) {
			b[i] = *new Matrix(outputs, 1);
		} else {
			b[i] = *new Matrix(neurons, 1);
		}
		randomDoubles(&b[i], MIN_WEIGHT, MAX_WEIGHT);
	}
	biases = b;
}