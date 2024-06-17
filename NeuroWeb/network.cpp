#include "network.h"

double moderlu(double x) {
    if (x < 0) {
        return 0; 
    }
    else if (x > 1) {
        return 1; 
    }
    else {
        return x; 
    }
}

double moderlu_derivative(double x) {
    if (x < 0 || x > 1) {
        return 0;
    }
    else {
        return 1;
    }
}


vector<double> matrix_multiply_sum(const vector<vector<double>>& A, const vector<double>& B, int input_neuro, int output_neuro, const vector<double>& bias) {
    vector<double> C;
    for (int i = 0; i < output_neuro; i++) {
        double temp = 0;
        for (int j = 0; j < input_neuro; j++) {
            temp += A[i][j] * B[j];
        }
        temp += bias[i];
        C.push_back(temp);
    }

    return C;
}

void MultilayerPerceptron::Init(int n)
{
    if (n == 0) {
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> dist(0.0, 1.0);

        for (int i = 0; i < hidden_size; i++) {
            vector<double> row;
            for (int j = 0; j < input_size; j++) {
                row.push_back(dist(gen));
            }
            weights_firstL.push_back(row);
        }
        for (int i = 0; i < output_size; i++) {
            vector<double> row;
            for (int j = 0; j < hidden_size; j++) {
                row.push_back(dist(gen));
            }
            weights_secondL.push_back(row);
        }
        for (int i = 0; i < hidden_size; i++) bias_firstL.push_back(dist(gen));
        for (int i = 0; i < output_size; i++) bias_secondL.push_back(dist(gen));
    }
    else {
        ifstream fw1("WEIGHTFIRST.txt");
        ifstream fw2("WEIGHTSECOND.txt");
        ifstream fb1("BFIRST.txt");
        ifstream fb2("BSECOND.txt");
        if (!fw1.is_open() || !fw2.is_open() || !fb1.is_open() || !fb2.is_open()) { cout << "Error file!\n"; return; }
        string s;
        for (int i = 0; i < hidden_size; i++) {
            vector<double> row;
            for (int j = 0; j < input_size; j++) {
                getline(fw2, s);
                const char* q = s.c_str();
                row.push_back(atof(q));
            }
            weights_firstL.push_back(row);
        }
        for (int i = 0; i < output_size; i++) {
            vector<double> row;
            for (int j = 0; j < hidden_size; j++) {
                getline(fw1, s);
                const char* c = s.c_str();
                row.push_back(atof(c));
            }
            weights_secondL.push_back(row);
        }
        for (int i = 0; i < hidden_size; i++) { getline(fb1, s); bias_firstL.push_back(stod(s)); }
        for (int i = 0; i < output_size; i++) {
            getline(fb2, s); bias_secondL.push_back(stod(s));
        }
    }
}

vector<double> MultilayerPerceptron::forward_propagation(const vector<double>& input, vector<double> &hidden_output)
{
    hidden_output = matrix_multiply_sum(weights_firstL, input, input_size, hidden_size, bias_firstL);
    for (int i = 0; i < hidden_output.size(); i++) hidden_output[i] = moderlu(hidden_output[i]);
    vector<double> output = matrix_multiply_sum(weights_secondL, hidden_output, hidden_size, output_size, bias_secondL);
    for (int i = 0; i < output.size(); i++) output[i] = moderlu(output[i]);
    return output;
}


void MultilayerPerceptron::backpropagation_updating_weights(const vector<double>& output, const vector<double>& target, double learning_rate, const vector<double>& input)
{
    int c = 0;
    /*ofstream f("weightstest.txt");
    ofstream fb("biastest.txt");*/
    vector<double> output_errors;
    for (int i = 0; i < output_size; i++) {
        output_errors.push_back((target[i] - output[i]) * moderlu_derivative(output[i]));
    }
    vector<double> hidden_errors = matrix_multiply(weights_secondL, output_errors, hidden_size, output_size, hidden_output);
    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < hidden_size; j++) {
            weights_secondL[i][j] += hidden_output[j] * output_errors[i] * learning_rate;
            /*if(weights_secondL[i][j] < 0) weights_secondL[i][j] = 0;
            if(weights_secondL[i][j] > 1) weights_secondL[i][j] = 1;*/
        }
        bias_secondL[i] += learning_rate * output_errors[i];
    }
    for (int i = 0; i < hidden_size; i++) {
        for (int j = 0; j < input_size; j++) {
            weights_firstL[i][j] += learning_rate * hidden_errors[i] * input[j];
            /*if (weights_firstL[i][j] < 0) weights_firstL[i][j] = 0;
            if (weights_firstL[i][j] > 1) weights_firstL[i][j] = 1;*/
        }
        bias_firstL[i] += learning_rate * hidden_errors[i];
    }
}


vector<double> MultilayerPerceptron::matrix_multiply(const vector<vector<double>>& A, const vector<double>& B, int input_neuro, int output_neuro, const vector<double>& input)
{
    vector<double> C;
    for (int i = 0; i < input_neuro; ++i) {
        double temp = 0;
        for (int j = 0; j < output_size; ++j) {
            temp += A[j][i] * B[j];
        }
        temp *= moderlu_derivative(input[i]);
        C.push_back(temp);
    }

    return C;
}

void MultilayerPerceptron::train(const vector<vector<double>>& inputs, const vector<vector<double>>& targets, int epochs, double learning_rate, string ans)
{
    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int i = 0; i < inputs.size(); i++) {
            vector<double> a = forward_propagation(inputs[i], hidden_output);
            backpropagation_updating_weights(a, targets[i], learning_rate, inputs[i]);
        }
        cout << "Epoch " << epoch + 1 << endl;
    }
    file(ans);
}

void MultilayerPerceptron::file(string ans)
{
    if (ans == "1") {
        ofstream fw1("WEIGHTFIRST.txt");
        ofstream fw2("WEIGHTSECOND.txt");
        ofstream fb1("BFIRST.txt");
        ofstream fb2("BSECOND.txt");
        if (!fw1.is_open() || !fw2.is_open() || !fb1.is_open() || !fb2.is_open()) { cout << "Error file!\n"; return; }
        for (auto& weight : weights_firstL) {
            for (int i = 0; i < weight.size(); i++) {
                fw1 << weight[i]; fw1 << endl;
            }
        }
        for (auto& weight : weights_secondL) {
            for (int i = 0; i < weight.size(); i++) {
                fw2 << weight[i]; fw2 << endl;
            }
        }
        for (int i = 0; i < bias_firstL.size(); i++) {
            fb1 << bias_firstL[i]; fb1 << endl;
        }
        for (int i = 0; i < bias_secondL.size(); i++) {
            fb2 << bias_secondL[i]; fb2 << endl;
        }
    }
    else if (ans == "2") {
        ofstream fw1("WEIGHTFIRSTTEST.txt");
        ofstream fw2("WEIGHTSECONDTEST.txt");
        ofstream fb1("BFIRSTTEST.txt");
        ofstream fb2("BSECONDTEST.txt");
        if (!fw1.is_open() || !fw2.is_open() || !fb1.is_open() || !fb2.is_open()) { cout << "Error file!\n"; return; }
        for (auto& weight : weights_firstL) {
            for (int i = 0; i < weight.size(); i++) {
                fw1 << weight[i]; fw1 << endl;
            }
        }
        for (auto& weight : weights_secondL) {
            for (int i = 0; i < weight.size(); i++) {
                fw2 << weight[i]; fw2 << endl;
            }
        }
        for (int i = 0; i < bias_firstL.size(); i++) {
            fb1 << bias_firstL[i]; fb1 << endl;
        }
        for (int i = 0; i < bias_secondL.size(); i++) {
            fb2 << bias_secondL[i]; fb2 << endl;
        }
    }
}


void clean() {
    cin.clear();
    cin.ignore((numeric_limits<streamsize>::max)(), '\n');
    cout << "Error!" << endl;
}

void read_inputs(vector<vector<double>>& inputs_or, vector<vector<double>>& inputs_and, vector<vector<double>>& inputs_xor) {
    ifstream inp("testinput.txt");
    string s, str;
    double w;
    const char* q;
    int c = 0;
    if (!inp.is_open()) { cout << "Error file!\n"; return; }
    while (getline(inp, s)){
        if (s == "inputs_or" || s == "inputs_and" || s == "inputs_xor") continue;
        else {
            vector<double> temp;
            stringstream ss(s);
            while (getline(ss, str, ' ')) {
                q = str.c_str();
                w = atof(q); temp.push_back(w);
            }
        c++;
        if (c <= 4) inputs_or.push_back(temp);
        else if(c>4 && c<=8) inputs_and.push_back(temp);
        else if(c>8) inputs_xor.push_back(temp);
        }
        
    }
}

void read_targets(vector<vector<double>>& targets_or, vector<vector<double>>& targets_and, vector<vector<double>>& targets_xor) {
    ifstream tar("target.txt");
    string s;
    double w;
    const char* q;
    int c = 0;
    if (!tar.is_open()) {
        cout << "Error file!\n";return;
    }
            while (getline(tar, s)) {
                if (s == "targets_or" || s == "targets_and" || s == "targets_xor") continue;
                else {
                    vector<double> temp;
                    q = s.c_str();
                    w = atof(q); temp.push_back(w);
                    c++;
                    if (c <= 4) targets_or.push_back(temp);
                    else if (c > 4 && c <= 8) targets_and.push_back(temp);
                    else if (c > 8) targets_xor.push_back(temp);
                }
        }
    }

void read_conf(vector<int> &config) {
    ifstream conf("testconfig.txt");
    if (!conf.is_open()) {
        cout << "Error file!\n"; return ;
    }
    string s, st;
    const char* q;
    int c;
    vector <int> temp;
    while (getline(conf, s)) {
        stringstream ss(s);
        while (getline(ss, st, ' ')) {
            q = st.c_str();
            c = stoi(q); temp.push_back(c);
        }
    }
    config = temp;
}

bool read_test(vector<int> &config, vector<vector<double>> &inputs_other, vector<vector<double>> & targets_other) {
    cout << "Образец файла для теста:\nconfig\nx y z\ninputs\na11 a12 a13...a1m\na21 a22 a23...a2m\n...........\nan1 an2 an3...anm";
    cout << "\noutputs\nb11 b12 b13...b1k\nb21 b22 b23...b2k\n...........\nbn1 bn2 bn3...bnk\n";
    cout << "\nГде x y z - количество нейронов первого, второго и третьего слоя\n{anm} - входные данные\n{onk} - результат\nВажно! Количество нейронов первого слоя (x) должно равняться количеству ";
    cout << "входных данных (m). Количество нейронов последнего слоя (z) должно равняться количеству выходных данных (k).\n\nПример:\ninputs\n1 1 1 1\noutputs\n1 1\nЗначит x = 4, z = 2\n\n";
    ifstream conf("test.txt");
    if (!conf.is_open()) {
        cout << "Error file!\n"; return false;
    }
    vector<vector<double>> inputs1_other;
    vector<vector<double>> outputs1_other;
    string s, st;
    vector<int> temp;
    int c = 0, k = 0;
    const char* q;
    int w;
    bool f = false; int u = 0;
    bool fi = false;
    while (getline(conf, s)) {
        if (c == 0 && s != "config") { cout << "Файл написан неверно! Обратите внимание на config\n"; return false; }
        else if (c == 0 && s == "config") {
            c++;  continue;
        }
        //проверка количества нейронов
        if (c == 1) {
            if (k == 3) {
                cout << "Слоёв всего три!\n"; return false;
            }
            if(s == "") {
                cout << "Что-то не так с количеством нейронов!\n"; return false;
            }
            stringstream ss(s);
            while (getline(ss, st, ' ')) {
                if (!(check(st))) { cout << "Количество нейронов должно быть числом (>0)\n"; return false; }
                q = st.c_str();
                w = stoi(q);
                if (w == 0) {
                    cout << "Количество нейронов должно быть числом (>0)\n"; return false;
                }
                temp.push_back(w);
                k++;
            }
        }
        if(k < 3) {
            cout << "Слоёв меньше трёх!\n"; return false;
        }
        if (c == 2 && s != "inputs") { cout << "Файл написан неверно! Обратите внимание на inputs\n"; return false; }
        else if (c == 2 && s == "inputs") {
            c++; fi = true; continue;
        }
        if ((c == 3 || c == 4 || c == 5) && s == "outputs") { cout << "Должно быть более двух примеров\n"; return false; }
        double w1;
        vector<double> tmp;
        stringstream sss(s);
        if (s != "outputs" && f == false && fi == true) {
            while (getline(sss, st, ' ')) {
                if (!(check(st))) { cout << "Входные данные - числа (только 1 и 0)\n"; return false; }
                q = st.c_str();
                w1 = atof(q);
                if (!(w1 == 0 || w1 == 1)) { cout << "Входные данные - числа (только 1 и 0)\n"; return false; }
                tmp.push_back(w1);
            }
            inputs1_other.push_back(tmp);
        }
        else if (s == "outputs") {
            c++; u++; f = true;
            if (u > 1) { cout << "Несколько раз outputs\n"; return false; }continue;
        }
        if (f == true && fi == true) {
            vector<double> tm;
            stringstream ssss(s);
            while (getline(ssss, st, ' ')) {
                if (!(check(st))) { cout << "Выходные данные - числа (только 1 и 0)\n"; return false; }
                q = st.c_str();
                w1 = atof(q);
                if (!(w1 == 0 || w1 == 1)) { cout << "Выходные данные - числа (только 1 и 0)\n"; return false; }
                tm.push_back(w1);
            }
            outputs1_other.push_back(tm);
        }
        else if(f == true && fi == false){
            cout << "В файле ошибка!\n"; return false;
        }
        c++;
    }
    if (f == false) {
        cout << "Нет outputs\n"; return false;
    }
    if (outputs1_other.size() != inputs1_other.size()) { cout << "Количество тестов входных и выходных данных должно быть одинаковым\n"; return false; }
    vector<double> temps;
    for (auto& input : inputs1_other) {
        int h = 0;
        for (int i = 0; i < input.size(); i++) {
            h++;
        }
        temps.push_back(h);
    }
    float sum = 0;
    for (int i = 0; i < temps.size(); i++) {
        if (temps[i] != temp[0]) { cout << " Количество нейронов первого слоя (x) должно равняться количеству входных данных (m).\n"; return false; }
        sum += temps[i];
    }
    if(sum/ temps.size() != temps[0]) { cout << " Количество чисел в inputs на каждой строке отличается.\n"; return false; }


    vector<double> te;
    for (auto& output : outputs1_other) {
        int h1 = 0;
        for (int i = 0; i < output.size(); i++) {
            h1++;
        }
        te.push_back(h1);
    }
    float sum1 = 0;
    for (int i = 0; i < te.size(); i++) {
        if (te[i] != temp[2]) { cout << " Количество нейронов последнего слоя (z) должно равняться количеству выходных данных (k).\n"; return false; }
        sum1 += te[i];
    }
    if (sum1 / te.size() != te[0]) { cout << " Количество чисел в outputs на каждой строке отличается.\n"; return false; }
    
    if(f==false) { cout << "Нет outputs\n"; return false; }
    config = temp; inputs_other = inputs1_other; targets_other = outputs1_other;
    return true;
}


void task_test(int v, double L, string ans, vector<vector<double>>& inputs_other, vector<vector<double>>& targets_other, MultilayerPerceptron& MLP) {
    MLP.Init(0);
    MLP.train(inputs_other, targets_other, v, L, ans);
    cout << "Результат тестирования:" << endl;
    int c = 0;
    double q;
    for (auto& input : inputs_other) {
        vector<double> output = MLP.forward_propagation(input, MLP.hidden_output);
        c = input.size();
        cout << "\nВход:\n";
        for (int i = 0; i < input.size(); i++) cout << input[i] << " ";
        cout << "\nВыход:\n";
        for (int i = 0; i < output.size(); i++) cout << round(output[i]) << " ";
        cout << endl;
    }
    vector <double>input_test;
    cout << "Введите входные данные (только 1 и 0) для своего задания\n";
    for (int i = 0; i < c; i++) {cin >> q; 
    while (!cin || cin.peek() != '\n' || !(q == 0 || q == 1)) {
        clean();
        cout << "только 1 и 0\n"; cin >> q;
    }
    input_test.push_back(q);
} 
    cout << "Результат:" << endl;
    vector<double> output = MLP.forward_propagation(input_test, MLP.hidden_output);
    cout << "\nВход:\n";
    for (int i = 0; i < input_test.size(); i++) cout << input_test[i] << " ";
    cout << "\nВыход:\n";
    for (int i = 0; i < output.size(); i++) cout << round(output[i]) << " ";
    cout << endl;
}



    void tasks(string a, int l, vector<vector<double>>&inputs_or, vector<vector<double>>&inputs_and,
        vector<vector<double>>&inputs_xor, vector<vector<double>>&targets_or,
        vector<vector<double>>&targets_and, vector<vector<double>>&targets_xor, MultilayerPerceptron &mlp, string ans)
{
    double q, w;
    if (a == "OR") {
        mlp.Init(l);
        mlp.train(inputs_or, targets_or, 1000, 0.01, ans);
        cout << "Результат тестирования:" << endl;
        for (auto& input : inputs_or) {
            vector<double> output = mlp.forward_propagation(input, mlp.hidden_output);
            cout << "Вход: " << input[0] << ", " << input[1] << " Выход: " << round(output[0]) << endl;
        }

        cout << "Введите два числа (только 1 и 0), к которым примените оператор OR\n"; cin >> q >> w;
        while (!cin || cin.peek() != '\n' || !(q == 0 || q == 1) || !(w == 0 || w == 1)) {
            clean();
            cout << "Введите два числа (только 1 и 0), к которым примените оператор OR\n"; cin >> q >> w;
        }
        cout << "Результат:" << endl;
        vector<double> input; input.push_back(q); input.push_back(w);
        vector<double> output = mlp.forward_propagation(input, mlp.hidden_output);
        cout << "Вход: " << input[0] << ", " << input[1] << " Выход: " << round(output[0]) << endl;
    }


    else if (a == "AND") {
        mlp.Init(l);
        mlp.train(inputs_and, targets_and, 1000, 0.01, ans);
        cout << "Результат тестирования:" << endl;
        for (auto& input : inputs_and) {
            vector<double> output = mlp.forward_propagation(input, mlp.hidden_output);
            cout << "Вход: " << input[0] << ", " << input[1] << " Выход: " << round(output[0]) << endl;
        }

        cout << "Введите два числа (только 1 и 0), к которым примените оператор AND\n"; cin >> q >> w;
        while (!cin || cin.peek() != '\n' || !(q == 0 || q == 1) || !(w == 0 || w == 1)) {
            clean();
            cout << "Введите два числа (только 1 и 0), к которым примените оператор AND\n"; cin >> q >> w;
        }
        cout << "Результат:" << endl;
        vector<double> input; input.push_back(q); input.push_back(w);
        vector<double> output = mlp.forward_propagation(input, mlp.hidden_output);
        cout << "Вход: " << input[0] << ", " << input[1] << " Выход: " << round(output[0]) << endl;
    }
    else if (a == "XOR") {
        mlp.Init(l);
        int k = 1000;
        if (l == 0) k = 7000;
        mlp.train(inputs_xor, targets_xor, k, 0.01, ans);
        cout << "Результат тестирования:" << endl;
        for (auto& input : inputs_xor) {
            vector<double> output = mlp.forward_propagation(input, mlp.hidden_output);
            cout << "Вход: " << input[0] << ", " << input[1] << " Выход: " << round(output[0]) << endl;
        }

        cout << "Введите два числа (только 1 и 0), к которым примените оператор XOR\n"; cin >> q >> w;
        while (!cin || cin.peek() != '\n' || !(q == 0 || q == 1) || !(w == 0 || w == 1)) {
            clean();
            cout << "Введите два числа (только 1 и 0), к которым примените оператор XOR\n"; cin >> q >> w;
        }
        cout << "Результат:" << endl;
        vector<double> input; input.push_back(q); input.push_back(w);
        vector<double> output = mlp.forward_propagation(input, mlp.hidden_output);
        cout << "Вход: " << input[0] << ", " << input[1] << " Выход: " << round(output[0]) << endl;
    }
}

bool check(string s) {
    return !s.empty() && std::find_if(s.begin(),
        s.end(), [](unsigned char c) { return !std::isdigit(c); }) == s.end();
}