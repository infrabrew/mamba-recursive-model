#!/usr/bin/env python3
"""
Data preparation and inspection script.
Use this to check what data will be loaded before training.
"""

import argparse
import os
from utils.data_loader import DataLoader


def main():
    parser = argparse.ArgumentParser(description="Prepare and inspect training data")
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing training files')
    parser.add_argument('--max_length', type=int, default=2048,
                        help='Maximum sequence length')
    parser.add_argument('--show_stats', action='store_true',
                        help='Show detailed statistics')
    parser.add_argument('--show_samples', type=int, default=0,
                        help='Number of sample chunks to display')

    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' does not exist")
        print(f"Creating directory '{args.data_dir}'...")
        os.makedirs(args.data_dir, exist_ok=True)
        print(f"\nPlease add your training files to '{args.data_dir}' and run again.")
        return

    print(f"Loading data from: {args.data_dir}")
    print(f"Maximum sequence length: {args.max_length}\n")

    # Load data
    loader = DataLoader(args.data_dir, args.max_length)
    documents = loader.load_directory()

    if not documents:
        print(f"No supported files found in '{args.data_dir}'")
        print("\nSupported formats:")
        print("  - Documents: .pdf, .txt, .md, .rst")
        print("  - Code: .py, .js, .ts, .java, .cpp, .c, .go, .rs, .rb, .php, etc.")
        print("  - Config: .json, .yaml, .yml, .toml, .ini, .xml")
        return

    print(f"Found {len(documents)} documents\n")

    # Prepare training chunks
    chunks = loader.prepare_training_data(documents)

    print(f"\nTotal training chunks: {len(chunks)}")

    if args.show_stats:
        print("\n" + "="*60)
        print("DETAILED STATISTICS")
        print("="*60)

        # File type statistics
        extensions = {}
        for doc in documents:
            ext = os.path.splitext(doc['source'])[1]
            extensions[ext] = extensions.get(ext, 0) + 1

        print("\nFiles by type:")
        for ext, count in sorted(extensions.items()):
            print(f"  {ext:10s}: {count:4d} files")

        # Document statistics
        total_chars = sum(len(doc['text']) for doc in documents)
        avg_doc_length = total_chars / len(documents)

        print(f"\nDocument statistics:")
        print(f"  Total documents: {len(documents)}")
        print(f"  Total characters: {total_chars:,}")
        print(f"  Avg chars/document: {avg_doc_length:,.0f}")

        # Chunk statistics
        total_chunk_chars = sum(len(chunk) for chunk in chunks)
        avg_chunk_length = total_chunk_chars / len(chunks)

        print(f"\nChunk statistics:")
        print(f"  Total chunks: {len(chunks)}")
        print(f"  Total characters: {total_chunk_chars:,}")
        print(f"  Avg chars/chunk: {avg_chunk_length:,.0f}")

        # Estimate training tokens (rough estimate: 1 token ≈ 4 chars)
        estimated_tokens = total_chunk_chars // 4
        print(f"  Estimated tokens: {estimated_tokens:,}")

    if args.show_samples > 0:
        print("\n" + "="*60)
        print(f"SAMPLE CHUNKS (showing {min(args.show_samples, len(chunks))})")
        print("="*60)

        for i in range(min(args.show_samples, len(chunks))):
            print(f"\nChunk {i + 1}:")
            print("-" * 60)
            print(chunks[i][:500] + "..." if len(chunks[i]) > 500 else chunks[i])
            print("-" * 60)

    print("\nData preparation complete!")
    print(f"\nTo train a model with this data, run:")
    print(f"  python train.py --data_dir {args.data_dir}")


if __name__ == '__main__':
    main()
