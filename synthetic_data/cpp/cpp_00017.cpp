#include <iostream>
#include <vector>
#include <algorithm>

template<typename T>
class Matrix {{
private:
    std::vector<std::vector<T>> data;
    size_t rows, cols;

public:
    Matrix(size_t r, size_t c) : rows(r), cols(c) {{
        data.resize(rows, std::vector<T>(cols, 0));
    }}

    T& operator()(size_t i, size_t j) {{
        return data[i][j];
    }}

    void print() const {{
        for (const auto& row : data) {{
            for (const auto& val : row) {{
                std::cout << val << " ";
            }}
            std::cout << std::endl;
        }}
    }}
}};

int main() {{
    Matrix<double> m(3, 3);
    m(0, 0) = 1.0;
    m.print();
    return 0;
}}