#include <iostream>

#include "srcnn.hpp"

using namespace std;

int main(int argc, char **argv)
{
    if(argc != 2)
    {
        cout << "Usage: ./SRCNN_cpp <input image path>" << endl;
        return 1;
    }
    string filename = argv[1];

    SRCNN srcnn;
    /*
    srcnn.checkWeightStatus();
    srcnn.generate(filename);
    srcnn.showOutput();
    */
    //srcnn.testConv();
    srcnn.testConv1Channel();
    //srcnn.generate(filename);
    //srcnn.showOutput();

    return 0;
}
