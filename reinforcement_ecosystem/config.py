#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Config module for the project
"""


import os


TF_LOG_DIR = '{}/../tf_logs'.format(os.path.dirname(os.path.realpath(__file__)))
CSV_LOG_DIR = '{}/../csv_logs'.format(os.path.dirname(os.path.realpath(__file__)))
PRINT_EACH = 0.1  # When to print the status of the algorithm
CSV_LOG_EACH = 0.01  # When to log the status of the algorithm into a CSV
