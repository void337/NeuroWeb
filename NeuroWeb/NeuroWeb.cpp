#include "network.h"
#include <iostream>
using namespace std;
int main()
{
    SetConsoleOutputCP(1251);
    SetConsoleCP(1251);
    string ans;
    while (true) {
        cout << "Обучить нейросеть готовым заданиям (OR, AND, XOR) - 1\nОбучить своему заданию - 2. Выйти из программы - 3\n"; cin >> ans;
        while (!(ans == "1" || ans == "2" || ans == "3")) { cout << "Обучить нейросеть готовыми заданиями (OR, AND, XOR) - 1\n Обучить своему заданию - 2. Выйти из программы - 3\n"; cin >> ans; }
        if (ans == "3") return 0;
        if (ans == "1") {

            vector<int> config;
            read_conf(config);
            string a; int l;
            MultilayerPerceptron mlp(config[0], config[1], config[2]);
            vector<vector<double>> inputs_or, inputs_and, inputs_xor;
            vector<vector<double>> targets_or, targets_and, targets_xor;
            read_inputs(inputs_or, inputs_and, inputs_xor);
            read_targets(targets_or, targets_and, targets_xor);

            cout << "Введите 0, если хотите начать с рандомными весами. Введите 1, если хотите начать c сохраненными весами. Введите 2, если хотите выйти из программы.\n"; cin >> l;
            while (!cin || cin.peek() != '\n' || !(l == 0 || l == 1 || l == 2)) {
                clean();
                cout << "Введите 0, если хотите начать с рандомных весов. Введите 1, если хотите начать c сохраненными весами. Введите 2, если хотите выйти из программы.\n"; cin >> l;
            }
            if (l == 2) return 0;
            cout << "Введите задачу для обучения:\nОперация OR, AND, XOR\n"; cin >> a;
            while (!(a == "OR" || a == "AND" || a == "XOR")) { cout << "Введите выбранную задачу (OR, AND, XOR)\n"; cin >> a; }
            tasks(a, l, inputs_or, inputs_and, inputs_xor, targets_or, targets_and, targets_xor, mlp, ans);
        }
        else {
            bool g = true;
            vector<vector<double>> inputs_other;
            vector<vector<double>> targets_other;
            vector<int> config1;
            bool fla = read_test(config1, inputs_other, targets_other);
            if (fla == false) {
                cout << "\nВы хотите заново ввести свое задание? - 1\nИли перейдёте к уже готовым заданиям? - 2\n";
                int k; cin >> k;
                while (!cin || cin.peek() != '\n' || !(k == 1 || k == 2)) {
                    clean();
                    cout << "Вы хотите заново ввести свое задание? - 1\nИли перейдёте к уже готовым заданиям? - 2\n"; cin >> k;
                }
                if (k == 2) continue;
                else {
                    cout << "Измените файл\n";
                    Sleep(3000);
                    fla = read_test(config1, inputs_other, targets_other);
                    while (fla == false) {
                        cout << "\nВы хотите заново ввести свое задание? - 1\nИли перейдёте к уже готовым заданиям? - 2\n";
                        int k; cin >> k;
                        while (!cin || cin.peek() != '\n' || !(k == 1 || k == 2)) {
                            clean();
                            cout << "Вы хотите заново ввести свое задание? - 1\nИли перейдёте к уже готовым заданиям? - 2\n"; cin >> k;
                        }
                        if (k == 2) {
                            fla = true; g = false;
                        }
                        else {
                            cout << "Измените файл\n";
                            Sleep(3000);
                            fla = read_test(config1, inputs_other, targets_other);
                        }

                    }
                }
            }
            if (g) {
                MultilayerPerceptron MLP(config1[0], config1[1], config1[2]);
                int v; double L;
                cout << "Введите количество эпох\n"; cin >> v;
                while (!cin || cin.peek() != '\n' || v <= 0) {
                    clean();
                    cout << "Введите количество эпох(>0)\n"; cin >> v;
                }
                cout << "Введите скорость обучения\n"; cin >> L;
                while (!cin || cin.peek() != '\n' || L <= 0) {
                    clean();
                    cout << "Введите скорость обучения(>0)\n"; cin >> L;
                }
                task_test(v, L, ans, inputs_other, targets_other, MLP);
            }
        }
    }
}