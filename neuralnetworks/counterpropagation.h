#include "matrix.h"

using namespace std;

/* CounterPropagation neural network written in C */

class CounterPropagation {
private:
    int N_INPUTS, N_KOHONEN, N_OUTPUTS;
    double MIN_WEIGHT, MAX_WEIGHT;
    Matrix k_weights, g_weights;
public:
    double ALPHA, BETA;
    CounterPropagation(int, int, int, double, double);
    Matrix online(Matrix, Matrix);
    Matrix eval(Matrix);
    void train(Matrix, Matrix);
};

/* ------------------------------------------------------
 * Methods associated with the "CounterPropagation" class
 * ------------------------------------------------------ */

// Constructor
CounterPropagation::CounterPropagation(int inputs, int kohonen, int outputs, double alpha, double beta) {
    N_INPUTS = inputs;
    N_KOHONEN = kohonen;
    N_OUTPUTS = outputs;
    ALPHA = alpha;
    BETA = beta;
    
    MIN_WEIGHT = 0;
    MAX_WEIGHT = 1;
    
    k_weights = *new Matrix(kohonen, inputs);
    g_weights = *new Matrix(outputs, kohonen);
    
    randomDoubles(&k_weights, MIN_WEIGHT, MAX_WEIGHT);
    randomDoubles(&g_weights, MIN_WEIGHT, MAX_WEIGHT);
}

// Both evaluate and run
Matrix CounterPropagation::online(Matrix input, Matrix output) {
    
    Matrix v1 = input.times(k_weights);
    
    // Find arg max
    int arg_max = 0;
    double max = v1.get(0, 0);
    for (int i = 1; i < v1.width; i++) {
        if (v1.get(i, 0) > max) {
            max = v1.get(i, 0);
            arg_max = i;
        }
    }
    
    // Update Kohonen weights to be closer to the input vector
    for (int i = 0; i < k_weights.height; i++) {
        double diff = k_weights.get(arg_max, i) + ALPHA * (input.get(i, 0) - k_weights.get(arg_max, i));
        k_weights.set(arg_max, i, diff);
    }
    
    // Get output of the network
    Matrix v2 = *new Matrix(g_weights.width, 1);
    for (int i = 0; i < g_weights.width; i++) {
        v2.set(i, 0, g_weights.get(i, arg_max));
    }
    
    // Update weights according to training rule
    for (int i = 0; i < g_weights.width; i++) {
        double diff = g_weights.get(i, arg_max) + BETA * (output.get(i, 0) - g_weights.get(i, arg_max));
        g_weights.set(i, arg_max, diff);
    }
    
    return v2;
}

// Evaluation method
Matrix CounterPropagation::eval(Matrix input) {
    Matrix v1 = input.times(k_weights);
    
    // Find arg max
    int arg_max = 0;
    double max = v1.get(0, 0);
    for (int i = 1; i < v1.width; i++) {
        if (v1.get(i, 0) > max) {
            max = v1.get(i, 0);
            arg_max = i;
        }
    }
    
    // Get the output data
    Matrix r = *new Matrix(g_weights.width, 1);
    for (int i = 0; i < g_weights.width; i++) {
        r.set(i, 0, g_weights.get(i, arg_max));
    }
    
    return r;
}

// Training method
void CounterPropagation::train(Matrix input, Matrix output) {
    Matrix v1 = input.times(k_weights);
    
    // Find arg max
    int arg_max = 0;
    double max = v1.get(0, 0);
    for (int i = 1; i < v1.width; i++) {
        if (v1.get(i, 0) > max) {
            max = v1.get(i, 0);
            arg_max = i;
        }
    }
    
    // Update Kohonen weights to be closer to the input vector
    for (int i = 0; i < k_weights.height; i++) {
        double diff = k_weights.get(arg_max, i) + ALPHA * (input.get(i, 0) - k_weights.get(arg_max, i));
        k_weights.set(arg_max, i, diff);
    }
    
    // Get output of the network
    Matrix v2 = *new Matrix(g_weights.width, 1);
    for (int i = 0; i < g_weights.width; i++) {
        v2.set(i, 0, g_weights.get(i, arg_max));
    }
    
    // Update weights according to training rule
    for (int i = 0; i < g_weights.width; i++) {
        double diff = g_weights.get(i, arg_max) + BETA * (output.get(i, 0) - g_weights.get(i, arg_max));
        g_weights.set(i, arg_max, diff);
    }
}