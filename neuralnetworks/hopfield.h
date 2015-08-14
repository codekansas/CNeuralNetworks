#include <iostream>
#include <vector>
#include "bmatrix.h"

using namespace std;

class Hopfield {
private:
    int N_INPUTS;
    Matrix weights;
public:
    // Training the matrix
    double getWeight(int, int, vector<Matrix>);
    void train(vector<Matrix>);
    void train(Matrix);
    
    // Run the network
    Matrix run(Matrix, int);
    Matrix runOnce(Matrix);
    Matrix calculateOutputs(Matrix);
    
    // Set and get weights
    void setWeights(Matrix);
    Matrix getWeights();
    
    // Constructor
    Hopfield(int inputs);
};

double Hopfield::getWeight(int i, int j, vector<Matrix> patterns) {
    const double num_patterns = patterns.size();
    
    double sum = 0;
    
    for (int n = 0; n < num_patterns; n++) {
        sum += (2 * patterns[n].get(i, 0) - 1) * (2 * patterns[n].get(j, 0) - 1);
    }
    
    return sum;
}

// Calculate the weights for the given neuron
void Hopfield::train(vector<Matrix> patterns) {
    for (int i = 0; i < N_INPUTS; i++) {
        for (int j = 0; j < N_INPUTS; j++) {
            weights.set(i, j, getWeight(i, j, patterns));
        }
    }
}

// Train on single pattern
void Hopfield::train(Matrix pattern) {
    for (int i = 0; i < N_INPUTS; i++) {
        for (int j = 0; j < N_INPUTS; j++) {
            weights.set(i, j, (2 * pattern.get(i, 0) - 1) * (2 * pattern.get(j, 0) - 1));
        }
    }
}

// Run the matrix
Matrix Hopfield::run(Matrix pattern, int max_iterations) {
    Matrix m = pattern.copy();
    bool changed = true;
    for (int i = 0; i < max_iterations && changed; i++) {
        pattern = m.copy();
        m = calculateOutputs(pattern);
        
        // Check if the matrix changed
        changed = false;
        for (int j = 0; j < m.width; j++) {
            if (m.get(j, 0) != pattern.get(j, 0)) {
                changed = true;
                break;
            }
        }
    }
    return pattern;
}

// Calculate output of a given neuron
Matrix Hopfield::calculateOutputs(Matrix input) {
    Matrix n = input.times(weights);
    for (int i = 0; i < n.width; i++) {
        if (n.get(i, 0) > 0) {
            n.set(i, 0, 1);
        } else {
            n.set(i, 0, -1);
        }
    }
    return n;
}

// Update the weights matrix
void Hopfield::setWeights(Matrix n_weights) {
    if (n_weights.width == N_INPUTS && n_weights.height == N_INPUTS) {
        weights = n_weights;
    }
}

// Get the weights matrix
Matrix Hopfield::getWeights() {
    return weights;
}

// Initialize a new network
Hopfield::Hopfield(int inputs) {
    N_INPUTS = inputs;
    weights = *randomDoubles(new Matrix(inputs, inputs), -1.0, 1.0);
}