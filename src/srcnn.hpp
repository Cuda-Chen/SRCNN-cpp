#ifndef SRCNN_HPP
#define SRCNN_HPP

#include <string>

class SRCNN
{
public:
    SRCNN();
    void generate(std::string filename);
private:
    int scale = 2;
};

#endif
