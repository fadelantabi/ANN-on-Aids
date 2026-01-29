
#include <fstream>
#include <string>
#include <iostream>
#include <vector>
#include <set>
#include <map>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <cctype>
#include <random>
#include <iomanip>
#include <algorithm>

using namespace std;
double sigmoid(double x)
{


	return 1.0 / (1.0 + exp((-x)));
}

double sigmoid_derivative(double x) {
	return x * (1.0 - x);
}
class NeuralNetwork {
public:
	vector<vector<double>> weights_input_hidden;
	vector<vector<double>> weights_hidden_output;
	double learning_rate;

	NeuralNetwork(int input_nodes, int hidden_nodes, int output_nodes, double lr) {
		learning_rate = lr;
		weights_input_hidden = random_weights(input_nodes + 1, hidden_nodes);
		weights_hidden_output = random_weights(hidden_nodes + 1, output_nodes);
	}

	vector<vector<double>> random_weights(int num_input, int num_output) {
		vector<vector<double>> weights(num_output,vector<double>(num_input));
		mt19937 gen(random_device{}());


		double range = 2.4 / (num_input-1);


		uniform_real_distribution<double> dist(-range, range);
		for (auto &row : weights) {
			for (auto &val : row) {
				 val = dist(gen);

			}


		}

		return weights;
	}

	vector<double> forward(vector<double> inputs, int stop_point) {
		vector<double> hidden_outputs;
		for (auto &weights : weights_input_hidden) {
			double total = 0.0;
			for (size_t i = 0; i < inputs.size() + 1; ++i) {
				if (i == inputs.size())
				{
					total += double(weights[i]);

				}
				else
				{
					total += (((double)inputs[i]) * weights[i]);
				}


			}
			hidden_outputs.push_back(sigmoid(total));

		}
	vector<double> final_outputs;
		for (auto &weights : weights_hidden_output) {
			double total = 0.0;
			for (size_t i = 0; i < hidden_outputs.size() + 1; ++i) {
				if (i == hidden_outputs.size())
				{
					total += double(weights[i]);

				}
				else
				{

					total += hidden_outputs[i] * weights[i];
				}


			}
			final_outputs.push_back(sigmoid(total));
		}

		if (stop_point == 1)
		{
			return hidden_outputs;

		}
		return final_outputs;
	}
	void backward(vector<double> inputs,vector<double> targets, vector<double> outputs) {
	vector<double> output_errors;
		for (size_t i = 0; i < targets.size(); ++i) {
			output_errors.push_back(((double)targets[i]) - outputs[i]);
		}
		vector<double> output_gradients;
		for (size_t i = 0; i < outputs.size(); ++i) {
			output_gradients.push_back(output_errors[i] * sigmoid_derivative(outputs[i]));
		}
		vector<vector<double>> weights_hidden_output2 = weights_hidden_output;;
		vector<double> hidden_outputs = forward(inputs, 1);
		for (size_t i = 0; i < weights_hidden_output.size(); ++i) {
			for (size_t j = 0; j < weights_hidden_output[i].size(); ++j) {
				if (weights_hidden_output[i].size() - 1 == j)
				{
					weights_hidden_output2[i][j] += (learning_rate * output_gradients[i]);
				}
				else
				{

					weights_hidden_output2[i][j] += (learning_rate * output_gradients[i] * hidden_outputs[j]);
				}


			}
		}

		vector<double> hidden_gradients;


		for (size_t i = 0; i < hidden_outputs.size(); ++i) {
			for (int j = 0; j < output_gradients.size(); j++)
			{

				hidden_gradients.push_back(sigmoid_derivative(hidden_outputs[i])*output_gradients[j] * weights_hidden_output[j][i]);
			}

		}
		weights_hidden_output = weights_hidden_output2;
		weights_hidden_output2.clear();
		for (size_t i = 0; i < weights_input_hidden.size(); ++i) {
			for (size_t j = 0; j < weights_input_hidden[i].size(); ++j)
			{
				if (j == weights_input_hidden[i].size() - 1)
				{
					weights_input_hidden[i][j] += (learning_rate * hidden_gradients[i]);
				}
				else
				{
					weights_input_hidden[i][j] += (learning_rate * hidden_gradients[i] * inputs[j]);
				}
			}
		}

	}


	void train(vector<double> inputs, vector<double> targets) {

		vector<double> outputs = forward(inputs, 0);

		backward(inputs, targets, outputs);
	}


	void test(vector<vector<double>> test_Aids_input, vector<vector<double>>test_Aids_output)
	{

		double error = 0;
		double error2 = 0;

		for (int i = 0; i < test_Aids_input.size(); i++)
		{

			vector<double> actual_output = forward(test_Aids_input[i], 0);
			error += ((test_Aids_output[i][0] - actual_output[0])*(test_Aids_output[i][0] - actual_output[0]));
			error2 += abs((test_Aids_output[i][0] - round(actual_output[0])));
		}
		cout <<"mse= "<< error / test_Aids_input.size() << endl;
		cout <<" percentage:  "<< (1-(error2 / test_Aids_input.size()))*100 << endl;
	}
};

int main() {
	random_device rd;
	mt19937 g(rd());
	vector<vector<double>> Aids_features;
	vector<vector<double>> Aids_infected;
	srand(static_cast<unsigned>(time(0)));
	ifstream myfile;
	myfile.open("AIDS2.txt");

	if (myfile.is_open()) {
		string line;

		while (getline(myfile, line)) {
			vector<double> Aids;
			int cnt2 = 0;
			size_t pos = 0;
			while ((pos = line.find(',')) != string::npos) {
				cnt2++;
				string token = line.substr(0, pos);
				Aids.push_back(stoi(token));
				line.erase(0, pos + 1);
			}
			Aids.push_back(stod(line));
			Aids_features.push_back(Aids);

		}

		myfile.close();


	}

	vector<double> mx1(Aids_features[0].size(), -1000000);
	vector<double> mn1(Aids_features[0].size(), 10000000);
	for (auto x : Aids_features)
		for (int i = 0; i < x.size(); i++)
		{
			mx1[i] = max(mx1[i], (double)x[i]);
			mn1[i] = min(mn1[i], (double)x[i]);
		}
	vector<vector<double>> Aids_features_dbl(Aids_features.size(), vector<double>(Aids_features[0].size(), 0.0));
	for (int i = 0; i < Aids_features_dbl.size(); i++)
		for (int j = 0; j < Aids_features_dbl[i].size(); j++)
		{
			Aids_features_dbl[i][j] = (((double)Aids_features[i][j] - mn1[j]) / (mx1[j] - mn1[j]));

		}

	Aids_features.clear();
	Aids_features = Aids_features_dbl;

	NeuralNetwork nn(22, 14, 1, 0.01);

	shuffle(Aids_features.begin(), Aids_features.end(), g);


	vector<vector<double>> Aids_features_temp;
	//for (int i = 0; i < Aids_features.size(); i++)
//{
	//	Aids_features_temp.push_back(Aids_features[i]);
	//}
	//Aids_features = Aids_features_temp;
	//Aids_features_temp.clear();
	vector<vector<double>> Aids_features_test;
	int noninfected = 0;
	for (int i = 0; i < Aids_features.size(); i++)
	{
		if (Aids_features[i].back() == 0.0)
		{
			noninfected++;

		}

	}
	int j = noninfected;
	noninfected = 0;
	for (int i = 0; i < Aids_features.size(); i++)
	{

		if (Aids_features[i].back() == 0.0 && noninfected <= Aids_features.size() - j)
		{

			noninfected++;
			Aids_features_temp.push_back(Aids_features[i]);

		}
		else if (Aids_features[i].back() != 0.0)
		{
			Aids_features_temp.push_back(Aids_features[i]);

		}

	}

	Aids_features = Aids_features_temp;
	Aids_features_temp.clear();
	shuffle(Aids_features.begin(), Aids_features.end(), g);
	int sze = Aids_features.size();
	for (int i = 0; i < sze/5; i++)
	{
		Aids_features_test.push_back(Aids_features.back());
		Aids_features.pop_back();
	}
	double current_mse = 0;
	double pre_mse = 0;
	double convergence_theshold = 1e-8;
	int max_epoch = 10000;
	for (int epoch = 0; epoch < max_epoch; ++epoch) {

		shuffle(Aids_features.begin(), Aids_features.end(), g);
		Aids_features_temp = Aids_features;
		Aids_infected.clear();
		for (size_t i = 0; i < Aids_features.size(); i++)
		{
			Aids_infected.push_back({ Aids_features[i].back() });
			Aids_features[i].pop_back();
		}
		for (size_t i = 0; i < Aids_features.size(); ++i) {
				nn.train(Aids_features[i], Aids_infected[i]);

		}
		pre_mse = current_mse;
		for (size_t i = 0; i < Aids_features.size(); ++i) {
			vector<double> mse_calculation=nn.forward(Aids_features[i],0);
			current_mse += pow(mse_calculation[0] - Aids_infected[i][0], 2);

		}
		current_mse /= (Aids_features.size());
		if (abs(current_mse - pre_mse) < convergence_theshold)
		{
			cout << "converged after epoch :" << epoch+1<<"  with loss "<< abs(current_mse - pre_mse)<<endl;
			break;
		}
		if (epoch + 1 == max_epoch)
		{
			cout << "max epoch reached" << endl;

		}
		//cout << " loss=" << current_mse<<endl;
		vector<vector<double>>Aids_features_temp_test = Aids_features_test;
		Aids_infected.clear();
		for (size_t i = 0; i < Aids_features_test.size(); i++)
		{
			Aids_infected.push_back({ Aids_features_test[i].back() });
			Aids_features_test[i].pop_back();
		}
		nn.test(Aids_features_test, Aids_infected);
		Aids_features_test = Aids_features_temp_test;

		Aids_features = Aids_features_temp;
	}

	Aids_infected.clear();
	for (size_t i = 0; i < Aids_features_test.size(); i++)
	{

		Aids_infected.push_back({ Aids_features_test[i].back() });
		Aids_features_test[i].pop_back();
	}
	nn.test(Aids_features_test, Aids_infected);

	return 0;
}
