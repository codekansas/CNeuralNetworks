#include "matrix.h"

class Network {
private:
    int N_INPUTS, N_OUTPUTS, H_SIZE, SEQ_LEN, ALPHA;
    Matrix Wxh, Whh, Why, Bh, By;
public:
    // Constructor
    Network(int inputs, int hidden, int outputs, int s_length, int l_rate) {
        
        // Initialize network properties
        N_INPUTS = inputs;
        H_SIZE = hidden;
        N_OUTPUTS = outputs;
        SEQ_LEN = s_length;
        ALPHA = l_rate;
        
        // Intialize matrices
        Wxh = *new Matrix(hidden, inputs);
        Whh = *new Matrix(hidden, hidden);
        Why = *new Matrix(outputs, hidden);
        
        randomDoubles(&Wxh, 0, 0.01);
        randomDoubles(&Whh, 0, 0.01);
        randomDoubles(&Why, 0, 0.01);
        
        // Initialize biases
        Bh = *new Matrix(hidden, 1);
        By = *new Matrix(outputs, 1);
    }
    
    // TODO finish this
};