#include <string>
#include <memory>
#include <stdexcept>

class DataProcessor {{
private:
    std::unique_ptr<int[]> buffer;
    size_t size;

public:
    DataProcessor(size_t n) : size(n) {{
        buffer = std::make_unique<int[]>(n);
    }}

    void process(const int* input, size_t len) {{
        if (len > size) {{
            throw std::runtime_error("Input too large");
        }}
        for (size_t i = 0; i < len; ++i) {{
            buffer[i] = input[i] * 2;
        }}
    }}

    int get(size_t index) const {{
        if (index >= size) {{
            throw std::out_of_range("Index out of range");
        }}
        return buffer[index];
    }}
}};