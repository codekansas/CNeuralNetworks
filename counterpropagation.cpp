#include <time.h>
#include "counterpropagation.h"

using namespace std;

// Main method
int main() {
	// Initialize randomizer
	srand(time(NULL));
	
	// Network parameters
	int N_INPUTS = 3;
	int N_KOHONEN = 5;
	int N_OUTPUTS = 2;
	double ALPHA = 0.05;
	double BETA = 0.05;
	
	// Set up network
	CounterPropagation* network = new CounterPropagation(N_INPUTS, N_KOHONEN, N_OUTPUTS, ALPHA, BETA);
	
	// Positive input
	double pos_in_data[3] = {0, 1, 0};
	Matrix* pos_in = new Matrix(N_INPUTS, 1, pos_in_data);
	
	// Negative input
	double neg_in_data[3] = {1, 0, 0};
	Matrix* neg_in = new Matrix(N_INPUTS, 1, neg_in_data);
	
	// Positive output
	double pos_out_data[2] = {0, 1};
	Matrix* pos_out = new Matrix(N_OUTPUTS, 1, pos_out_data);
	
	// Negative output
	double neg_out_data[2] = {1, 0};
	Matrix* neg_out = new Matrix(N_OUTPUTS, 1, neg_out_data);
	
	// Positive training
	for (int i = 0; i < 1000; i++) {
		network->train(pos_in, pos_out);
	}
	
	// Negative training
	for (int i = 0; i < 1000; i++) {
		network->train(neg_in, neg_out);
	}
	
	// Test
	network->eval(pos_in)->print();
	network->eval(neg_in)->print();
}