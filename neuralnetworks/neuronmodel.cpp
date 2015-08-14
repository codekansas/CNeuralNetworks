#include "neuronmodel.h"

// For writing to files
#include <fstream>

int test() {
    // Open file
    ofstream output_file;
    output_file.open("/Users/judgingmoloch/Desktop/output.txt");
    
    // Simulation parameters
    double dt = 0.001; // time step, in milliseconds
    
    // Run neuron model
    Simulation *sim = new Simulation(dt);
    sim->addNeurons(1);
    
    sim->reset();
    sim->setInput(0, 60);
    int n = 10;
    for (int i = 0; i < n; i++) {
        sim->startCountingSpikes();
        for (int j = 0; j < 100 / dt; j++) {
            sim->step();
            sim->print(output_file);
        }
        cout << i+1 << "/" << n << endl;
//        output_file << i << "\t";
//        sim->printCounts(output_file);
    }
    
    // Close file
    output_file.close();
    
    return 0;
}
