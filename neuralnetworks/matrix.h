#include <iostream>

using namespace std;

class Matrix {
private:
    double* data; // Pointer to data
    bool transpose;
    void init();
public:
    int width, height;
    Matrix();
    Matrix(int, int);
    Matrix(int, int, double*);
    double get(int, int);
    void set(int, int, double);
    string getVals();
    void print();
    Matrix T();
    Matrix reshape(int, int);
    Matrix times(Matrix);
    Matrix dot(Matrix);
    Matrix times(double);
    Matrix minus(Matrix);
    Matrix plus(Matrix);
    Matrix plus(double);
    double sum();
    Matrix copy();
    double* getData();
};

/* ------------------------------------------
 * Methods associated with the "Matrix" class
 * ------------------------------------------ */

// Constructors
Matrix::Matrix() {
    init();
    width = 0;
    height = 0;
}
Matrix::Matrix(int w, int h) {
    init();
    width = w;
    height = h;
    data = new double[w * h];
}
Matrix::Matrix(int w, int h, double *d) {
    init();
    width = w;
    height = h;
    data = d;
}
void Matrix::init() {
    transpose = false;
}

// Access methods
double Matrix::get(int x, int y) {
    if (transpose) {
        return data[x * height + y];
    } else {
        return data[y * width + x];
    }
}

void Matrix::set(int x, int y, double v) {
    if (transpose) {
        data[x * height + y] = v;
    } else {
        data[y * width + x] = v;
    }
}

double* Matrix::getData() {
    return data;
}

// Print methods
string Matrix::getVals() {
    string s = "";
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            s += to_string(get(j, i)) + "\t";
        }
        s += "\n";
    }
    return s;
}
void Matrix::print() {
    cout << getVals() << endl;
}

Matrix Matrix::T() {
    Matrix r = *new Matrix(height, width, data);
    r.transpose = true;
    return r;
}

Matrix Matrix::reshape(int w, int h) {
    Matrix r = *new Matrix(w, h, data);
    return r;
}

// Matrix multiplication
Matrix Matrix::times(Matrix m) {
    if (width != m.height) {
        return *new Matrix(0, 0);
    }
    Matrix r = *new Matrix(m.width, height);
    
    // Multiplication
    for (int i = 0; i < m.width; i++) {
        for (int j = 0; j < height; j++) {
            double t = 0;
            for (int k = 0; k < width; k++) {
                t += get(k, j) * m.get(i, k);
            }
            r.set(i, j, t);
        }
    }
    
    return r;
}

// Element-wise multiplication
Matrix Matrix::dot(Matrix m) {
    if (width != m.width || height != m.height) {
        return *new Matrix(0, 0);
    }
    Matrix r = *new Matrix(width, height);
    
    // Multiplication
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            r.set(i, j, get(i, j) * m.get(i, j));
        }
    }
    
    return r;
}

Matrix Matrix::times(double d) {
    Matrix n = *new Matrix(width, height);
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            n.set(i, j, get(i, j) * d);
        }
    }
    return n;
}

Matrix Matrix::minus(Matrix m) {
    if (width != m.width || height != m.height) {
        return *new Matrix(0, 0);
    }
    Matrix r = *new Matrix(width, height);
    
    // Addition
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            r.set(i, j, get(i, j) - m.get(i, j));
        }
    }
    
    return r;
}

Matrix Matrix::plus(Matrix m) {
    if (width != m.width || height != m.height) {
        return *new Matrix(0, 0);
    }
    Matrix r = *new Matrix(width, height);
    
    // Addition
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            r.set(i, j, get(i, j) + m.get(i, j));
        }
    }
    
    return r;
}

Matrix Matrix::plus(double d) {
    Matrix m = *new Matrix(width, height);
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            m.set(i, j, get(i, j) + d);
        }
    }
    return m;
}

double Matrix::sum() {
    double t = 0;
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            t += get(i, j);
        }
    }
    return t;
}

Matrix Matrix::copy() {
    Matrix m = *new Matrix(width, height);
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            m.set(i, j, get(i, j));
        }
    }
    return m;
}

// Some methods for randomizing matrices
double randn(double lo, double hi);
int randi(int lo, int hi);
Matrix* randomDoubles(Matrix *m, double lo, double hi);
Matrix* randomInts(Matrix *m, int lo, int hi);

// Generate a random number between <lo> and <hi>
double randn(double lo, double hi) {
    return lo + ((double) rand() / RAND_MAX) * (hi - lo);
}

double randn() {
    return (double) rand() / RAND_MAX;
}

// Generate a random integer between <lo> and <hi>
int randi(int lo, int hi) {
    return rand() % (hi - lo) + lo;
}

// Randomize the contents of a matrix to values between <lo> and <hi>
Matrix* randomDoubles(Matrix *m, double lo, double hi) {
    for (int i = 0; i < m->width; i++) {
        for (int j = 0; j < m->height; j++) {
            m->set(i, j, randn(lo, hi));
        }
    }
    return m;
}

// Randomize the contents of a matrix with random integer values
Matrix* randomInts(Matrix *m, int lo, int hi) {
    for (int i = 0; i < m->width; i++) {
        for (int j = 0; j < m->height; j++) {
            m->set(i, j, randi(lo, hi));
        }
    }
    return m;
}