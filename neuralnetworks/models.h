#include <iostream>
#include <vector>
#include <map>

using namespace std;

/*
 A random mix of neuron models
 */

// TODO Morris-Lecar

class LIFNeuron {
private:
    double v_mem = 0.0, ref = 0.0, *dt, last_spike_time = 0.0, t_rc = 0.02, t_ref = 0.002, t_pstc = 0.1;
    int sim_steps = 0, n_outputs = 0;
    map<int,double> input_voltages;
    
    vector<LIFNeuron*> inputs;
    
    vector<double> output_weights;
    vector<double> output_delays;
    vector<LIFNeuron*> outputs;
    bool spiked = false;
public:
    // Constructor
    LIFNeuron(double *sim_dt) {
        dt = sim_dt;
    }
    
    // Some accessor methods
    double getLastSpikeTime() {
        return last_spike_time;
    }
    double getVoltage() {
        return v_mem;
    }
    void printVoltage(ostream &stream) {
        stream << v_mem << "\t";
    }
    
    // Synapse dynamics
    void addVoltage(double amount, double at_time) {
        if (input_voltages.count(at_time)) {
            input_voltages[(int) (at_time / *dt)] += amount;
        } else {
            input_voltages[(int) (at_time / *dt)] = amount;
        }
    }
    
    double getInitialWeight() {
        return 1.0;
    }
    
    double getDelay() {
        return 0.01;
    }
    
    // Connect onto another neuron
    void onto(LIFNeuron *n) {
        n->inputs.push_back(this);
        
        outputs.push_back(n);
        output_delays.push_back(getDelay());
        output_weights.push_back(getInitialWeight());
        n_outputs++;
    }
    
    // The bulk of the simulation dynamics
    void step() {
        sim_steps++;
        v_mem += *dt * (input_voltages[sim_steps] - v_mem) / t_rc;
        input_voltages.erase(sim_steps); // Delete this element
        
        // Set membrane voltage to at least zero
        if (v_mem < 0) {
            v_mem = 0;
        }
        
        // Refractory period
        if (ref > 0) {
            v_mem = 0;
            ref -= *dt;
        }
        
        // Spike
        if (v_mem > 1) {
            last_spike_time = sim_steps * *dt;
            spiked = true;
            v_mem = 0;
            ref = t_ref;
            
            // Fire onto another neuron
            for (int i = 0; i < n_outputs; i++) {
                outputs[i]->addVoltage(1.0 * output_weights[i], last_spike_time + output_delays[i]);
            }
        } else {
            spiked = false;
        }
        
        // TODO - STDP dynamics
    }
};