#!/usr/bin/env python3
"""
Script to run the PXA analysis from the project root directory.
"""
import os
import sys

# Add the code directory to the Python path
code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'code')
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

# Import the main function from the pxa module
from pxa import main

if __name__ == "__main__":
    # Run the main analysis function
    main() 