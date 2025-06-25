#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 14:51:01 2025

@author: vincentcalia-bogan

util exclusively for loading the object "standardized_chagnepoints_dict" -- an antequated but entrenched data format
also so that intermediate files can continue to be saved.
"""
from typing import Dict, Any
from pathlib import Path
import pickle

def load_standardized_changepoints(pkl_dir: str) -> Dict[str, Any]:
    """
    Scan pkl_dir for files named "<clean_name>_changepoints.pkl",
    load each one, strip off the "_changepoints" suffix from the stem,
    and return a dict clean_name -> loaded array.

    Example:
        >>> d = load_standardized_changepoints("/.../RNN_PROCESSING_PARQUETS/changepoints")
        >>> list(d.keys())[:2]
        ['AM35_4Tastes_201229_150307', 'AM25_4Tastes_200806_094914']
    """
    pkl_dir = Path(pkl_dir)
    cp_dict: Dict[str, Any] = {}

    for pkl_path in pkl_dir.glob("*_changepoints.pkl"):
        clean_name = pkl_path.stem.rsplit("_changepoints", 1)[0]
        with open(pkl_path, "rb") as f:
            cp_dict[clean_name] = pickle.load(f)
    return cp_dict