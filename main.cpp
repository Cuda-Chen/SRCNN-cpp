#include <iostream>

#include "srcnn.hpp"

using namespace std;

int main(int argc, char **argv)
{
    string filename = argv[1];

    SRCNN srcnn;
    srcnn.generate(filename);

    return 0;
}
