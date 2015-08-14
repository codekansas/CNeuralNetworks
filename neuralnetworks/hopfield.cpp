#include "hopfield.h"

void normalize(Matrix *m) {
    for (int i = 0; i < m->width; i++) {
        for (int j = 0; j < m->height; j++) {
            if (m->get(i, j) < 0) {
                m->set(i, j, -1);
            } else {
                m->set(i, j, 1);
            }
        }
    }
}

Matrix getRandVector(int n) {
    Matrix v = *new Matrix(n, 1);
    
    for (int i = 0; i < n; i++) {
        v.set(i, 0, randn(-1, 1));
    }
    
    normalize(&v);
    
    return v;
}

Matrix noisy(Matrix a, double noise) {
    Matrix b = *new Matrix(a.width, a.height);
    
    for (int i = 0; i < a.width; i++) {
        for (int j = 0; j < a.height; j++) {
            b.set(i, j, a.get(i, j) + randn(-noise, noise));
        }
    }
    
    normalize(&b);
    
    return b;
}

int main() {
    srand((unsigned int) time(NULL));
    
    Hopfield *net = new Hopfield(16);
    
    Matrix m = *new Matrix(16, 1, (double[]) {
        1,  1,  1,  1,
        1,  -1, -1, 1,
        1,  -1, -1, 1,
        1,  1,  1,  1,
    });
    
    Matrix n = *new Matrix(16, 1, (double[]) {
        1,  -1, -1, 1,
        -1, 1,  1,  -1,
        -1, 1,  1,  -1,
        1,  -1, -1, 1,
    });
    
    vector<Matrix> v(2);
    v[0] = n;
    v[1] = m;
    
    net->train(v);
    
    for (int i = 0; i < 10; i++) {
        m = noisy(m, 1.5);
        
        cout << "Before:" << endl;
        m.reshape(4,4).print();
        
        cout << "After:" << endl;
        net->run(m, 100).reshape(4,4).print();
    }
    
    return 0;
}
