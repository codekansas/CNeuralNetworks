#include <time.h>
#include "counterpropagation.h"

using namespace std;

// Main method
int main() {
	// Initialize randomizer
	srand(time(NULL));
	
	// Network parameters
	int N_INPUTS = 3;
	int N_KOHONEN = 500;
	int N_OUTPUTS = 2;
	double ALPHA = 0.5;
	double BETA = 0.5;
	
	// Set up network
	CounterPropagation network = *new CounterPropagation(N_INPUTS, N_KOHONEN, N_OUTPUTS, ALPHA, BETA);
	
	// Data
	double in1[3] = { 0, 0, 1 };
	double in2[3] = { 0, 1, 1 };
	double in3[3] = { 1, 0, 1 };
	double in4[3] = { 1, 1,  };
	
	double out1[1] = { 0 };
	double out2[1] = { 1 };
	
	Matrix m_in1 = *new Matrix(N_INPUTS, 1, in1);
	Matrix m_in2 = *new Matrix(N_INPUTS, 1, in2);
	Matrix m_in3 = *new Matrix(N_INPUTS, 1, in3);
	Matrix m_in4 = *new Matrix(N_INPUTS, 1, in4);
	
	Matrix m_out1 = *new Matrix(N_OUTPUTS, 1, out1);
	Matrix m_out2 = *new Matrix(N_OUTPUTS, 1, out2);
	
	cout << "Before training:" << endl;
	network.eval(m_in1).print();
	network.eval(m_in2).print();
	network.eval(m_in3).print();
	network.eval(m_in4).print();
	
	int N = 100;
	for (int i = 0; i < N; i++) {
		network.train(m_in1, m_out1);
		network.train(m_in2, m_out2);
		network.train(m_in3, m_out2);
		network.train(m_in4, m_out1);
		network.ALPHA -= ALPHA / (1.1 * N);
		network.BETA -= BETA / (1.1 * N);
	}
	
	// Test
	cout << "After training:" << endl;
	network.eval(m_in1).print();
	network.eval(m_in2).print();
	network.eval(m_in3).print();
	network.eval(m_in4).print();
    
    return 0;
}