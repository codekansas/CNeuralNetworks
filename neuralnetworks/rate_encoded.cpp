#include "models.h"
#include <fstream>

int main() {
    double dt = 0.001;
    
    LIFNeuron *n = new LIFNeuron(&dt);
    LIFNeuron *m = new LIFNeuron(&dt);
    
    n->onto(m);
    
    // Output file to write data to
    ofstream output_file;
    output_file.open("/Users/judgingmoloch/Desktop/output.txt");
    
    for (int i = 0; i < 1000; i++) {
        if (i % 5 == 0) {
            n->addVoltage(2, i * dt);
        } else {
            n->addVoltage(1.1, i * dt);
        }
    }
    
    for (int i = 0; i < 1000; i++) {
        n->step();
        m->step();
        output_file << (i * dt) << "\t";
        n->printVoltage(output_file);
        m->printVoltage(output_file);
        output_file << endl;
    }
    
}