#include <vector>
#include <time.h>
#include "matrix.h"

// Classes defined in this file
class Synapse;
class Neuron;
class Soma;

// Simulation Class
// ----------------

class Simulation {
protected:
    vector<Neuron*> neurons;
    vector<double> inputs;
    int n_neurons = 0;
    
    // For counting spikes
    int *counts;
    double started_counting_spikes = 0;
public:
    double dt;
    double sim_time;
    void addNeurons(int);
    void setInput(int, double);
    void step();
    void print(ostream&);
    void printCounts(ostream&);
    void printRates(ostream&);
    
    // For counting spikes
    void startCountingSpikes();
    int* getSpikeCounts();
    double getTimeSinceStartedCounting();
    
    // Reset simulation
    void reset();
    
    Simulation(double);
};

// Neuron Class
// ------------

class Neuron {
protected:
    Soma *soma;
    Simulation *sim;
public:
    // Record and count spikes
    bool spiked = false;
    int spike_count = 0;
    
    Neuron(Simulation*);
    void step(double);
    double getVoltage();
    void reset();
};

// Soma Class
// ----------

class Soma {
protected:
    // Model parameters
    double v_peak = 35;
    double v_threshold = -40;
    double v_reset = -60;
    double k = 0.7;
    double mem_cap = 100;
    double bias = 0;
    
    // Cortical pyramidal neurons
    double a = 0.03;
    double b = -2;
    double c = -50;
    double d = 100;
    
    // Parents
    Neuron *parent;
    Simulation *sim;
    
    // Extras
    double dt;
public:
    // Dynamic variables
    double u = 0;
    double v = v_reset;
    
    // Izhikevich neuron model
    void step(double);
    void reset();
    
    // Constructor
    Soma(Neuron*, Simulation*);
};

// Simulation Methods
// ------------------

double getInput() {
    return randn(40, 80);
}

// For counting spikes
void Simulation::startCountingSpikes() {
    counts = new int[n_neurons];
    for (int i = 0; i < n_neurons; i++) {
        counts[i] = neurons[i]->spike_count;
    }
    started_counting_spikes = sim_time;
}
int* Simulation::getSpikeCounts() {
    for (int i = 0; i < n_neurons; i++) {
        counts[i] += neurons[i]->spike_count;
    }
    return counts;
}
double Simulation::getTimeSinceStartedCounting() {
    return sim_time - started_counting_spikes;
}

void Simulation::addNeurons(int n) {
    srand((unsigned int) time(NULL));
    n_neurons += n;
    
    // Reserve space for neurons
    neurons.reserve(n_neurons);
    inputs.reserve(n_neurons);
    
    // Add neurons and inputs
    for (int i = 0; i < n; i++) {
        neurons.push_back(new Neuron(this));
        inputs.push_back(getInput());
    }
}

void Simulation::reset() {
    for (int i = 0; i < n_neurons; i++) {
        neurons[i]->reset();
    }
}

void Simulation::setInput(int w, double v) {
    inputs[w] = v;
}

Simulation::Simulation(double time_step) {
    dt = time_step;
}

void Simulation::print(ostream &stream) {
    stream << sim_time << "\t";
    for (int i = 0; i < n_neurons; i++) {
        stream << neurons[i]->getVoltage() << "\t";
    }
    stream << endl;
}

void Simulation::printCounts(ostream &stream) {
    for (int i = 0; i < n_neurons; i++) {
        stream << neurons[i]->spike_count - counts[i] << "\t";
    }
    stream << endl;
}

void Simulation::printRates(ostream &stream) {
    for (int i = 0; i < n_neurons; i++) {
        stream << (neurons[i]->spike_count - counts[i]) / getTimeSinceStartedCounting() << "\t";
    }
    stream << endl;
}

void Simulation::step() {
    sim_time += dt;
    for (int i = 0; i < n_neurons; i++) {
        neurons[i]->step(inputs[i]);
    }
}

// Neuron Methods
// --------------

Neuron::Neuron(Simulation *s) {
    soma = new Soma(this, s);
    sim = s;
}

void Neuron::reset() {
    // Reset dynamic variables
    soma->reset();
    
    // Reset spike counting
    spike_count = 0;
    spiked = false;
}

void Neuron::step(double input) {
    soma->step(input);
}

double Neuron::getVoltage() {
    return soma->v;
}

// Soma Methods
// ------------

Soma::Soma(Neuron *p, Simulation *s) {
    // Remember parent neuron
    parent = p;
    sim = s;
    
    // Get time step
    dt = sim->dt;
}

void Soma::reset() {
    v = v_reset;
    u = 0;
}

void Soma::step(double input) {
    v += (k * (v - v_reset) * (v - v_threshold) - u + input + bias) * dt / mem_cap;
    u += (a * (b * (v - v_reset) - u)) * dt;
    
    // Spike dynamics
    if (v > v_peak) {
        v = c;
        u += d;
        parent->spiked = true;
        parent->spike_count++;
    } else {
        parent->spiked = false;
    }
}

