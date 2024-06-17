#pragma once
#ifndef NETWORK_H
#define NETWORK_H
#include <iostream>
#include <random>
#include <windows.h>
#include <fstream>
#include <string>
#include <vector>
#include <ctime>
#include <cmath>
#include <sstream>
#include <cstdlib>

using namespace std;

double moderlu(double x);
double moderlu_derivative(double x);
vector<double> matrix_multiply_sum(const vector<vector<double>>& A, const vector<double>& B, int input_neuro, int output_neuro, const vector<double>& bias);

class MultilayerPerceptron {
	int input_size;
	int hidden_size;
	int output_size;
	vector<vector<double>> weights_firstL;
	vector<vector<double>> weights_secondL;
	vector<double> bias_firstL;
	vector<double> bias_secondL;


public:
	vector<double> hidden_output;
	MultilayerPerceptron(int input_size, int hidden_size, int output_size) {
		this->input_size = input_size; this->hidden_size = hidden_size; this->output_size = output_size;
	};
	void Init(int n);
	vector<double> forward_propagation(const vector<double>& input, vector<double>& hidden_output);
	void backpropagation_updating_weights(const vector<double>& output, const vector<double>& target, double learning_rate, const vector<double>& input);
	vector<double> matrix_multiply(const vector<vector<double>>& A, const vector<double>& B, int input_neuro, int output_neuro, const vector<double>& input);
	void train(const vector<vector<double>>& inputs, const vector<vector<double>>& targets, int epochs, double learning_rate, string ans);
	void file(string ans);
};

void clean();
void read_targets(vector<vector<double>>& targets_or, vector<vector<double>>& targets_and, vector<vector<double>>& targets_xor);
void tasks(string a, int l, vector<vector<double>>& inputs_or, vector<vector<double>>& inputs_and,
	vector<vector<double>>& inputs_xor, vector<vector<double>>& targets_or,
	vector<vector<double>>& targets_and, vector<vector<double>>& targets_xor, MultilayerPerceptron& mlp, string ans);
void read_conf(vector<int>& config);
void read_inputs(vector<vector<double>>& inputs_or, vector<vector<double>>& inputs_and, vector<vector<double>>& inputs_xor);
bool read_test(vector<int>& config, vector<vector<double>>& inputs_other, vector<vector<double>>& targets_other);
bool check(string s);
void task_test(int v, double L, string ans, vector<vector<double>>& inputs_other, vector<vector<double>>& targets_other, MultilayerPerceptron& MLP);

#endif