#include <iostream>
#include <vector>
#include <cmath>
using namespace std;

int main(int argc, char *argv[])
{
    int dimension = (argc > 1) ? stoi(argv[1]) : 1;
    vector<int> cores(12);
    for (int p = 1; p <= 12; ++p)
        cores[p-1] = dimension * pow(p, dimension +1);
    for(int i = 0; i < cores.size(); ++i)
        cout << "Running with " << cores[i] << " cores for dimension " << dimension << " and p = " << i + 1 << std::endl;
}