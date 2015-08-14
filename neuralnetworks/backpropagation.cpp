#include "backpropagation.h"

int backprop() {
    // Initialize randomizer
    srand((unsigned int) time(NULL));
    
    // Network parameters
    int N_LAYERS = 2;
    int NEURONS_PER_LAYER = 5;
    int N_INPUTS = 3;
    int N_OUTPUTS = 2;
    double LEARNING_RATE = 10;
    
    BackPropagation sim = *new BackPropagation(N_LAYERS, NEURONS_PER_LAYER, N_INPUTS, N_OUTPUTS, LEARNING_RATE);
    
    double in1[3] = { 0, 0, 1 };
    double in2[3] = { 0, 1, 1 };
    double in3[3] = { 1, 0, 1 };
    double in4[3] = { 1, 1, 1 };
    
    double out1[2] = { 0, 1 };
    double out2[2] = { 1, 0 };
    
    Matrix m_in1 = *new Matrix(N_INPUTS, 1, in1);
    Matrix m_in2 = *new Matrix(N_INPUTS, 1, in2);
    Matrix m_in3 = *new Matrix(N_INPUTS, 1, in3);
    Matrix m_in4 = *new Matrix(N_INPUTS, 1, in4);
    
    Matrix m_out1 = *new Matrix(N_OUTPUTS, 1, out1);
    Matrix m_out2 = *new Matrix(N_OUTPUTS, 1, out2);
    
    cout << "Before training:" << endl;
    sim.eval(m_in1).print();
    sim.eval(m_in2).print();
    sim.eval(m_in3).print();
    sim.eval(m_in4).print();
    
    int N = 1000;
    for (int i = 0; i < N; i++) {
        sim.train(m_in1, m_out1);
        sim.train(m_in2, m_out2);
        sim.train(m_in3, m_out2);
        sim.train(m_in4, m_out1);
        sim.LEARNING_RATE -= LEARNING_RATE / (2 * N);
    }
    
    cout << "After training:" << endl;
    sim.eval(m_in1).print();
    sim.eval(m_in2).print();
    sim.eval(m_in3).print();
    sim.eval(m_in4).print();
    
    return 0;
}