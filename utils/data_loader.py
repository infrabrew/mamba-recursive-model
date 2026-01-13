"""
Data loader utilities for loading various file formats for Mamba model training.
Supports: PDF, TXT, source code files, markdown, JSON, config files, and datasets.
"""

import os
import json
import glob
from pathlib import Path
from typing import List, Dict, Union, Generator
import re

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None


class DataLoader:
    """Load and preprocess data from various file formats."""

    SUPPORTED_CODE_EXTENSIONS = {
        '.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.cpp', '.c', '.h',
        '.cs', '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.scala',
        '.sh', '.bash', '.r', '.sql', '.html', '.css', '.vue', '.svelte'
    }

    SUPPORTED_TEXT_EXTENSIONS = {
        '.txt', '.md', '.markdown', '.rst', '.tex', '.log'
    }

    SUPPORTED_CONFIG_EXTENSIONS = {
        '.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf', '.xml'
    }

    def __init__(self, data_dir: str = "data", max_length: int = 2048):
        """
        Initialize DataLoader.

        Args:
            data_dir: Directory containing training data
            max_length: Maximum sequence length for tokenization
        """
        self.data_dir = data_dir
        self.max_length = max_length

    def load_pdf(self, file_path: str) -> str:
        """Load text from PDF file."""
        if PyPDF2 is None:
            raise ImportError("PyPDF2 is required for PDF loading. Install with: pip install PyPDF2")

        text = []
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text.append(page.extract_text())
        except Exception as e:
            print(f"Error loading PDF {file_path}: {e}")
            return ""

        return "\n".join(text)

    def load_text_file(self, file_path: str) -> str:
        """Load text from a text-based file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return ""

    def load_json(self, file_path: str) -> str:
        """Load and stringify JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Convert JSON to formatted string
                return json.dumps(data, indent=2)
        except Exception as e:
            print(f"Error loading JSON {file_path}: {e}")
            return ""

    def load_file(self, file_path: str) -> str:
        """Load a single file based on its extension."""
        ext = Path(file_path).suffix.lower()

        if ext == '.pdf':
            return self.load_pdf(file_path)
        elif ext == '.json':
            return self.load_json(file_path)
        else:
            # All other text-based files
            return self.load_text_file(file_path)

    def load_directory(self, directory: str = None) -> List[Dict[str, str]]:
        """
        Load all supported files from a directory.

        Args:
            directory: Directory to load from (defaults to self.data_dir)

        Returns:
            List of dictionaries with 'text' and 'source' keys
        """
        if directory is None:
            directory = self.data_dir

        all_extensions = (
            self.SUPPORTED_CODE_EXTENSIONS |
            self.SUPPORTED_TEXT_EXTENSIONS |
            self.SUPPORTED_CONFIG_EXTENSIONS |
            {'.pdf'}
        )

        documents = []

        for root, _, files in os.walk(directory):
            for file in files:
                ext = Path(file).suffix.lower()
                if ext in all_extensions:
                    file_path = os.path.join(root, file)
                    text = self.load_file(file_path)

                    if text.strip():  # Only add non-empty documents
                        documents.append({
                            'text': text,
                            'source': file_path
                        })
                        print(f"Loaded: {file_path} ({len(text)} chars)")

        return documents

    def load_huggingface_dataset(self, dataset_name: str, split: str = 'train',
                                  text_column: str = 'text') -> List[Dict[str, str]]:
        """
        Load a dataset from HuggingFace.

        Args:
            dataset_name: Name of the dataset on HuggingFace
            split: Dataset split to load
            text_column: Name of the column containing text
                         Can be comma-separated for instruction datasets (e.g., 'instruction,output')

        Returns:
            List of dictionaries with 'text' and 'source' keys
        """
        if load_dataset is None:
            raise ImportError("datasets library required. Install with: pip install datasets")

        print(f"Loading HuggingFace dataset: {dataset_name}")
        dataset = load_dataset(dataset_name, split=split)

        # Check if text_column is comma-separated (for instruction datasets)
        columns = [col.strip() for col in text_column.split(',')]

        documents = []
        for idx, example in enumerate(dataset):
            if len(columns) > 1:
                # Instruction-response format: combine multiple columns
                text_parts = []
                for col in columns:
                    if col in example and example[col]:
                        text_parts.append(str(example[col]))

                if text_parts:
                    # Format as instruction-response pair
                    text = '\n'.join(text_parts)
                    documents.append({
                        'text': text,
                        'source': f"{dataset_name}:{split}:{idx}"
                    })
            else:
                # Single column format
                if columns[0] in example:
                    documents.append({
                        'text': example[columns[0]],
                        'source': f"{dataset_name}:{split}:{idx}"
                    })

        print(f"Loaded {len(documents)} examples from {dataset_name}")
        return documents

    def chunk_text(self, text: str, chunk_size: int = None,
                   overlap: int = 128) -> List[str]:
        """
        Split text into chunks for training.

        Args:
            text: Text to chunk
            chunk_size: Size of each chunk (defaults to self.max_length)
            overlap: Overlap between chunks

        Returns:
            List of text chunks
        """
        if chunk_size is None:
            chunk_size = self.max_length

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap

        return chunks

    def prepare_training_data(self, documents: List[Dict[str, str]],
                             chunk_size: int = None) -> List[str]:
        """
        Prepare documents for training by chunking.

        Args:
            documents: List of document dictionaries
            chunk_size: Size of chunks

        Returns:
            List of text chunks ready for training
        """
        all_chunks = []

        for doc in documents:
            chunks = self.chunk_text(doc['text'], chunk_size)
            all_chunks.extend(chunks)

        print(f"Prepared {len(all_chunks)} training chunks from {len(documents)} documents")
        return all_chunks


class StreamingDataLoader:
    """Memory-efficient streaming data loader for large datasets."""

    def __init__(self, data_dir: str = "data", chunk_size: int = 2048):
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.loader = DataLoader(data_dir, chunk_size)

    def stream_files(self) -> Generator[str, None, None]:
        """Stream text chunks from files one at a time."""
        all_extensions = (
            self.loader.SUPPORTED_CODE_EXTENSIONS |
            self.loader.SUPPORTED_TEXT_EXTENSIONS |
            self.loader.SUPPORTED_CONFIG_EXTENSIONS |
            {'.pdf'}
        )

        for root, _, files in os.walk(self.data_dir):
            for file in files:
                ext = Path(file).suffix.lower()
                if ext in all_extensions:
                    file_path = os.path.join(root, file)
                    text = self.loader.load_file(file_path)

                    if text.strip():
                        # Yield chunks from this file
                        chunks = self.loader.chunk_text(text)
                        for chunk in chunks:
                            yield chunk
