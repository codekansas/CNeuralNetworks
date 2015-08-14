#include <iostream>
#include "matrix.h"

using namespace std;
using byte = unsigned char;

class BMatrix {
private:
    byte *data;
    bool transpose = false;
    int coords(int, int);
public:
    int width, height, n;
    BMatrix();
    BMatrix(int, int);
    BMatrix(int, int, byte*);
    bool isDifferent(BMatrix);
    void print();
    BMatrix T();
    Matrix times(Matrix);
    bool get(int, int);
    void set(int, int, bool);
    void randomize();
};

int BMatrix::coords(int x, int y) {
    if (transpose) {
        return y*width+x;
    } else {
        return x*height+y;
    }
}

bool BMatrix::get(int x, int y) {
    int elem = coords(x, y);
    return (data[elem/8] & (1 << (elem%8))) != 0;
}

void BMatrix::set(int x, int y, bool v) {
    int elem = coords(x, y);
    
    if (v) {
        data[elem/8] |= 1 << (elem%8);
    } else {
        data[elem/8] &= ~(1 << (elem%8));
    }
}

Matrix BMatrix::times(Matrix m) {
    if (width != m.height) {
        return *new Matrix(0, 0);
    }
    Matrix r = *new Matrix(m.width, height);
    
    // Multiplication
    for (int i = 0; i < m.width; i++) {
        for (int j = 0; j < height; j++) {
            double t = 0;
            for (int k = 0; k < width; k++) {
                if (get(k, j)) {
                    t += m.get(i, j);
                }
            }
            r.set(i, j, t);
        }
    }
    
    return r;
}

void BMatrix::print() {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            cout << get(j, i) << "\t";
        }
        cout << endl;
    }
}

bool BMatrix::isDifferent(BMatrix m) {
    if (m.width != width || m.height!= height) {
        return true;
    }
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            if (m.get(i, j) != get(i, j)) {
                return true;
            }
        }
    }
    return false;
}

BMatrix BMatrix::T() {
    BMatrix m = *new BMatrix(height, width, data);
    m.transpose = !transpose;
    return m;
}

BMatrix::BMatrix() {
    width = 0;
    height = 0;
    n = 0;
    data = new byte[1] { 0 };
}

BMatrix::BMatrix(int w, int h) {
    width = w;
    height = h;
    n = w*h/8+1;
    
    data = new byte[n];
    
    // Set all elements in array to zero
    for (int i = 0; i < n; i++) {
        data[i] = 0;
    }
}

BMatrix::BMatrix(int w, int h, byte *d) {
    width = w;
    height = h;
    n = w*h/8+1;
    
    data = d;
}