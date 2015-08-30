//
//  recurrent.cpp
//  neuralnetworks
//
//  Created by Benjamin Bolte on 8/16/15.
//  Copyright (c) 2015 Benjamin Bolte. All rights reserved.
//

#include "recurrent.h"

int main() {
    // Initialize random number generator
    srand((unsigned int) time(NULL));
    
    const int INPUTS = 2, HIDDEN = 3, OUTPUTS = 4, SEQ_LENGTH = 25, LEARNING_RATE = 0.1;
    
    Network *net = new Network(INPUTS, HIDDEN, OUTPUTS, SEQ_LENGTH, LEARNING_RATE);
    
    
    
    return 0;
}