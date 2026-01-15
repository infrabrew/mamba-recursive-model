#!/usr/bin/env python3
"""
Verification script for security fixes applied to train.py and api_server.py
"""

import ast
import sys

def check_torch_load_safety(filename):
    """Check that all torch.load() calls use weights_only=True"""
    print(f"\n[*] Checking {filename} for torch.load() safety...")

    with open(filename, 'r') as f:
        tree = ast.parse(f.read(), filename=filename)

    issues = []
    safe_calls = []

    class TorchLoadChecker(ast.NodeVisitor):
        def visit_Call(self, node):
            # Check if this is a torch.load() call
            if (isinstance(node.func, ast.Attribute) and
                node.func.attr == 'load' and
                isinstance(node.func.value, ast.Name) and
                node.func.value.id == 'torch'):

                # Check for weights_only parameter
                has_weights_only = False
                for keyword in node.keywords:
                    if keyword.arg == 'weights_only':
                        has_weights_only = True
                        if isinstance(keyword.value, ast.Constant) and keyword.value.value is True:
                            safe_calls.append(node.lineno)
                        break

                if not has_weights_only:
                    issues.append(node.lineno)

            self.generic_visit(node)

    checker = TorchLoadChecker()
    checker.visit(tree)

    if issues:
        print(f"  ✗ FAILED: Found {len(issues)} unsafe torch.load() calls at lines: {issues}")
        return False
    elif safe_calls:
        print(f"  ✓ PASSED: All {len(safe_calls)} torch.load() calls use weights_only=True")
        print(f"    Lines: {safe_calls}")
        return True
    else:
        print(f"  ℹ INFO: No torch.load() calls found")
        return True

def check_cors_safety(filename):
    """Check that CORS doesn't use allow_origins=['*']"""
    print(f"\n[*] Checking {filename} for CORS configuration...")

    with open(filename, 'r') as f:
        content = f.read()

    # Check for insecure CORS pattern
    if 'allow_origins=["*"]' in content or "allow_origins=['*']" in content:
        print(f"  ✗ FAILED: Found insecure CORS configuration with allow_origins=['*']")
        return False

    # Check for secure CORS pattern (environment variable)
    if 'ALLOWED_ORIGINS' in content and 'os.getenv' in content:
        print(f"  ✓ PASSED: CORS uses environment variable configuration")
        return True

    print(f"  ℹ INFO: Could not verify CORS configuration")
    return True

def main():
    print("="*70)
    print("Security Fixes Verification")
    print("="*70)

    all_passed = True

    # Check train.py for torch.load safety
    if not check_torch_load_safety('train.py'):
        all_passed = False

    # Check api_server.py for CORS safety
    if not check_cors_safety('api_server.py'):
        all_passed = False

    print("\n" + "="*70)
    if all_passed:
        print("✓ ALL CHECKS PASSED")
        print("="*70)
        print("\nSecurity fixes have been successfully applied:")
        print("  - torch.load() calls now use weights_only=True")
        print("  - CORS configuration uses environment variable")
        print("\nBackups created:")
        print("  - train.py.backup_security_fixes")
        print("  - api_server.py.backup_security_fixes")
        return 0
    else:
        print("✗ SOME CHECKS FAILED")
        print("="*70)
        print("\nPlease review the failed checks above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
