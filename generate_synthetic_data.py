#!/usr/bin/env python3
"""
Generate synthetic training data for Mamba model.
Creates realistic code files in Python (ML/DL/AI), Go, C++, and Ada.
"""

import os
import random
from pathlib import Path


class SyntheticCodeGenerator:
    """Generate realistic synthetic code files."""

    def __init__(self, output_dir: str = "synthetic_data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # Python ML/DL/AI Templates
    PYTHON_IMPORTS = [
        "import numpy as np",
        "import pandas as pd",
        "import torch",
        "import torch.nn as nn",
        "import torch.optim as optim",
        "from torch.utils.data import Dataset, DataLoader",
        "import tensorflow as tf",
        "from tensorflow import keras",
        "from sklearn.model_selection import train_test_split",
        "from sklearn.preprocessing import StandardScaler",
        "import matplotlib.pyplot as plt",
        "import seaborn as sns",
        "from transformers import AutoModel, AutoTokenizer",
        "import cv2",
        "from PIL import Image",
    ]

    PYTHON_ML_FUNCTIONS = [
        """def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    \"\"\"Train a neural network model.\"\"\"
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}')
    return model""",

        """def create_cnn_model(input_shape, num_classes):
    \"\"\"Create a Convolutional Neural Network.\"\"\"
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model""",

        """def preprocess_data(X, y, test_size=0.2):
    \"\"\"Preprocess and split dataset.\"\"\"
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler""",

        """class TransformerModel(nn.Module):
    \"\"\"Transformer model for sequence processing.\"\"\"
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 5000, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x) + self.pos_encoder[:, :x.size(1), :]
        x = self.transformer(x)
        return self.fc(x)""",
    ]

    PYTHON_ML_CLASSES = [
        """class CustomDataset(Dataset):
    \"\"\"Custom PyTorch Dataset.\"\"\"
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label""",

        """class NeuralNetwork(nn.Module):
    \"\"\"Simple feedforward neural network.\"\"\"
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x""",
    ]

    # Go Templates
    GO_TEMPLATES = [
        """package main

import (
    "fmt"
    "log"
    "net/http"
)

func main() {{
    http.HandleFunc("/", handler)
    log.Println("Server starting on :8080")
    log.Fatal(http.ListenAndServe(":8080", nil))
}}

func handler(w http.ResponseWriter, r *http.Request) {{
    fmt.Fprintf(w, "Hello, World!")
}}""",

        """package utils

import (
    "encoding/json"
    "io/ioutil"
)

type Config struct {{
    Host string `json:"host"`
    Port int    `json:"port"`
    Debug bool `json:"debug"`
}}

func LoadConfig(path string) (*Config, error) {{
    data, err := ioutil.ReadFile(path)
    if err != nil {{
        return nil, err
    }}

    var config Config
    err = json.Unmarshal(data, &config)
    return &config, err
}}""",

        """package database

import (
    "database/sql"
    _ "github.com/lib/pq"
)

type DB struct {{
    conn *sql.DB
}}

func NewDB(connStr string) (*DB, error) {{
    conn, err := sql.Open("postgres", connStr)
    if err != nil {{
        return nil, err
    }}
    return &DB{{conn: conn}}, nil
}}

func (db *DB) Query(query string, args ...interface{{}}) (*sql.Rows, error) {{
    return db.conn.Query(query, args...)
}}

func (db *DB) Close() error {{
    return db.conn.Close()
}}""",
    ]

    # C++ Templates
    CPP_TEMPLATES = [
        """#include <iostream>
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
}}""",

        """#include <string>
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
}};""",

        """#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>

template<typename T>
class ThreadSafeQueue {{
private:
    mutable std::mutex mutex;
    std::queue<T> queue;
    std::condition_variable cond;

public:
    void push(T value) {{
        std::lock_guard<std::mutex> lock(mutex);
        queue.push(std::move(value));
        cond.notify_one();
    }}

    bool try_pop(T& value) {{
        std::lock_guard<std::mutex> lock(mutex);
        if (queue.empty()) {{
            return false;
        }}
        value = std::move(queue.front());
        queue.pop();
        return true;
    }}

    void wait_and_pop(T& value) {{
        std::unique_lock<std::mutex> lock(mutex);
        cond.wait(lock, [this] {{ return !queue.empty(); }});
        value = std::move(queue.front());
        queue.pop();
    }}
}};""",
    ]

    # Ada Templates
    ADA_TEMPLATES = [
        """with Ada.Text_IO; use Ada.Text_IO;

procedure Hello is
begin
   Put_Line("Hello, World!");
end Hello;""",

        """package Matrix_Operations is
   type Matrix is array (Positive range <>, Positive range <>) of Float;

   function Multiply(A, B : Matrix) return Matrix;
   function Transpose(M : Matrix) return Matrix;
   procedure Print(M : Matrix);
end Matrix_Operations;

package body Matrix_Operations is
   function Multiply(A, B : Matrix) return Matrix is
      Result : Matrix(A'Range(1), B'Range(2));
   begin
      for I in A'Range(1) loop
         for J in B'Range(2) loop
            Result(I, J) := 0.0;
            for K in A'Range(2) loop
               Result(I, J) := Result(I, J) + A(I, K) * B(K, J);
            end loop;
         end loop;
      end loop;
      return Result;
   end Multiply;

   function Transpose(M : Matrix) return Matrix is
      Result : Matrix(M'Range(2), M'Range(1));
   begin
      for I in M'Range(1) loop
         for J in M'Range(2) loop
            Result(J, I) := M(I, J);
         end loop;
      end loop;
      return Result;
   end Transpose;

   procedure Print(M : Matrix) is
   begin
      for I in M'Range(1) loop
         for J in M'Range(2) loop
            Put(Float'Image(M(I, J)) & " ");
         end loop;
         New_Line;
      end loop;
   end Print;
end Matrix_Operations;""",

        """with Ada.Containers.Vectors;

package Data_Structures is
   type Element is record
      ID : Integer;
      Value : Float;
      Name : String(1..50);
   end record;

   package Element_Vectors is new Ada.Containers.Vectors
     (Index_Type   => Natural,
      Element_Type => Element);

   procedure Add_Element(V : in out Element_Vectors.Vector; E : Element);
   function Find_By_ID(V : Element_Vectors.Vector; ID : Integer) return Element;
end Data_Structures;

package body Data_Structures is
   procedure Add_Element(V : in out Element_Vectors.Vector; E : Element) is
   begin
      V.Append(E);
   end Add_Element;

   function Find_By_ID(V : Element_Vectors.Vector; ID : Integer) return Element is
   begin
      for E of V loop
         if E.ID = ID then
            return E;
         end if;
      end loop;
      raise Constraint_Error with "Element not found";
   end Find_By_ID;
end Data_Structures;""",
    ]

    def generate_python_file(self, index: int) -> str:
        """Generate a Python ML/DL/AI script."""
        imports = random.sample(self.PYTHON_IMPORTS, random.randint(3, 7))
        functions = random.sample(self.PYTHON_ML_FUNCTIONS, random.randint(1, 3))
        classes = random.sample(self.PYTHON_ML_CLASSES, random.randint(0, 2))

        content = '"""Machine Learning Module {}."""\n\n'.format(index)
        content += '\n'.join(imports) + '\n\n'

        for cls in classes:
            content += cls + '\n\n'

        for func in functions:
            content += func + '\n\n'

        content += f"""
if __name__ == '__main__':
    print("Running ML script {index}...")
    # Example usage
    model = create_model()
    train_data = load_data()
    results = train_model(model, train_data)
    print(f"Training complete: {{results}}")
"""
        return content

    def generate_go_file(self, index: int) -> str:
        """Generate a Go script."""
        template = random.choice(self.GO_TEMPLATES)
        return template

    def generate_cpp_file(self, index: int) -> str:
        """Generate a C++ script."""
        template = random.choice(self.CPP_TEMPLATES)
        return template

    def generate_ada_file(self, index: int) -> str:
        """Generate an Ada script."""
        template = random.choice(self.ADA_TEMPLATES)
        return template

    def generate_files(self, lang: str, count: int, extension: str):
        """Generate multiple files for a language."""
        lang_dir = os.path.join(self.output_dir, lang)
        os.makedirs(lang_dir, exist_ok=True)

        generator_map = {
            'python': (self.generate_python_file, '.py'),
            'go': (self.generate_go_file, '.go'),
            'cpp': (self.generate_cpp_file, '.cpp'),
            'ada': (self.generate_ada_file, '.adb'),
        }

        generator, ext = generator_map.get(lang, (None, extension))
        if not generator:
            print(f"Unknown language: {lang}")
            return

        print(f"Generating {count} {lang.upper()} files...")
        for i in range(count):
            filename = f"{lang}_{i+1:05d}{ext}"
            filepath = os.path.join(lang_dir, filename)

            content = generator(i + 1)

            with open(filepath, 'w') as f:
                f.write(content)

            if (i + 1) % 100 == 0:
                print(f"  Generated {i+1}/{count} files...")

        print(f"✓ Completed {count} {lang.upper()} files in {lang_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic training data")
    parser.add_argument('--output', type=str, default='synthetic_data',
                        help='Output directory')
    parser.add_argument('--python', type=int, default=1000,
                        help='Number of Python ML/DL/AI scripts')
    parser.add_argument('--go', type=int, default=100,
                        help='Number of Go scripts')
    parser.add_argument('--cpp', type=int, default=500,
                        help='Number of C++ scripts')
    parser.add_argument('--ada', type=int, default=2000,
                        help='Number of Ada scripts')

    args = parser.parse_args()

    print("=" * 60)
    print("Synthetic Data Generator")
    print("=" * 60)
    print(f"\nOutput directory: {args.output}")
    print(f"Python scripts: {args.python}")
    print(f"Go scripts: {args.go}")
    print(f"C++ scripts: {args.cpp}")
    print(f"Ada scripts: {args.ada}")
    print(f"Total: {args.python + args.go + args.cpp + args.ada} files")
    print()

    generator = SyntheticCodeGenerator(args.output)

    # Generate files
    generator.generate_files('python', args.python, '.py')
    generator.generate_files('go', args.go, '.go')
    generator.generate_files('cpp', args.cpp, '.cpp')
    generator.generate_files('ada', args.ada, '.adb')

    print("\n" + "=" * 60)
    print("Generation Complete!")
    print("=" * 60)
    print(f"\nFiles created in: {args.output}/")
    print(f"\nYou can now train with:")
    print(f"  python train.py --data_dir {args.output} --model_size medium --vram 16gb")


if __name__ == '__main__':
    main()
