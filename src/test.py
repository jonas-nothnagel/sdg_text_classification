#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for bashing on server
"""

import pandas as pd
import joblib
from torch import cuda

if __name__ == '__main__':

    device = 'cuda' if cuda.is_available() else 'cpu'
    print(device)
    
    print("test this script.....")

    print("")

    print("it works!!")

    print("")