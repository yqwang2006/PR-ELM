#ifndef POSELM_H
#define POSELM_H
#include "armadillo"
#include <string.h>
using namespace arma;

#define LABELDIM 1
#define DATADIM 36+1

double ELM_test(mat TestData, mat TestLabels, mat *InputWeight, mat OutputWeight, mat *Bias, std::string *ActivationFunc, double C, int layers, int NumberClasses);

void run_ELM(mat TrainData,mat TrainLabels, mat &OutputWeight, mat &H_H, mat *InputWeight, mat *Bias, std::string *ActivationFunction, double C, int layers, int NumberClasses );

void combine_model(mat &combine_weight, mat &combine_HH, mat combine_HT, double C);

mat onehot(int rows, int cols, mat labels);

mat ActivateFunction(std::string ActivationFunction, mat x);

mat sigmoid(const mat x);

mat tanh(const mat x);

mat rectifier(const mat x);

mat linear(const mat x);

void read_dat_file ( mat &Data, mat &Labels, std::string filename, int data_dim );
#endif
