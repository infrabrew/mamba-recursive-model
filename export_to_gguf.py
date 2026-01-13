#!/usr/bin/env python3
"""
Export trained Mamba model to GGUF format for Ollama and LM Studio.

This requires llama.cpp's convert.py script.
Install with: pip install gguf
"""

import argparse
import os
import sys


def export_to_gguf(hf_model_dir: str, output_file: str, quantization: str = "Q4_K_M"):
    """
    Export HuggingFace model to GGUF format.

    Args:
        hf_model_dir: Path to HuggingFace model directory
        output_file: Path to output GGUF file
        quantization: Quantization type (Q4_K_M, Q5_K_M, Q8_0, etc.)
    """
    print("=" * 60)
    print("GGUF Export Process")
    print("=" * 60)

    # Check if gguf is installed
    try:
        import gguf
        print("✓ gguf package found")
    except ImportError:
        print("✗ gguf package not found")
        print("\nPlease install with:")
        print("  pip install gguf")
        sys.exit(1)

    # Check if llama.cpp is available
    llama_cpp_convert = "llama.cpp/convert.py"  # Update this path

    if not os.path.exists(llama_cpp_convert):
        print(f"\n✗ llama.cpp not found at: {llama_cpp_convert}")
        print("\nTo convert to GGUF format, you need llama.cpp:")
        print("\n1. Clone llama.cpp:")
        print("   git clone https://github.com/ggerganov/llama.cpp.git")
        print("\n2. Run the conversion script:")
        print(f"   python llama.cpp/convert.py {hf_model_dir} --outfile {output_file}")
        print("\n3. (Optional) Quantize the model:")
        print(f"   ./llama.cpp/quantize {output_file} {output_file.replace('.gguf', f'-{quantization}.gguf')} {quantization}")
        print("\nFor now, the HuggingFace model has been exported.")
        print("Follow the steps above to complete GGUF conversion.")
        return

    # If llama.cpp is available, run conversion
    import subprocess

    print(f"\nConverting {hf_model_dir} to GGUF...")
    print(f"Output: {output_file}")

    try:
        # Convert to GGUF (FP16)
        cmd = [
            "python",
            llama_cpp_convert,
            hf_model_dir,
            "--outfile", output_file,
            "--outtype", "f16"
        ]

        print(f"\nRunning: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        print(f"\n✓ Converted to GGUF (FP16): {output_file}")

        # Quantize if requested
        if quantization and quantization != "f16":
            quantized_file = output_file.replace('.gguf', f'-{quantization}.gguf')
            quantize_cmd = [
                "./llama.cpp/quantize",
                output_file,
                quantized_file,
                quantization
            ]

            print(f"\nQuantizing to {quantization}...")
            print(f"Running: {' '.join(quantize_cmd)}")
            subprocess.run(quantize_cmd, check=True)
            print(f"\n✓ Quantized model: {quantized_file}")

            return quantized_file

        return output_file

    except subprocess.CalledProcessError as e:
        print(f"\n✗ Conversion failed: {e}")
        print("\nManual conversion steps:")
        print(f"1. python llama.cpp/convert.py {hf_model_dir} --outfile {output_file}")
        print(f"2. ./llama.cpp/quantize {output_file} {output_file.replace('.gguf', f'-{quantization}.gguf')} {quantization}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Export Mamba model to GGUF format")
    parser.add_argument('--hf_model', type=str, required=True,
                        help='Path to HuggingFace model directory (use export_to_huggingface.py first)')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output GGUF file')
    parser.add_argument('--quantization', type=str, default='Q4_K_M',
                        choices=['Q4_K_M', 'Q5_K_M', 'Q8_0', 'f16'],
                        help='Quantization type (default: Q4_K_M)')

    args = parser.parse_args()

    print("\nIMPORTANT: Make sure you've exported to HuggingFace format first:")
    print(f"  python export_to_huggingface.py --checkpoint checkpoints/final --output {args.hf_model}")
    print()

    export_to_gguf(args.hf_model, args.output, args.quantization)


if __name__ == '__main__':
    main()
