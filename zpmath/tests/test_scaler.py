# -*- coding: utf-8 -*-
#
# Copyright (C) 2023 Zeroth-Principles
#
# This file is part of Zeroth-Meta.
#
#  Zeroth-Meta is free software: you can redistribute it and/or modify it under the
#  terms of the GNU General Public License as published by the Free Software
#  Foundation, either version 3 of the License, or (at your option) any later
#  version.
# 
#  Zeroth-Meta is distributed in the hope that it will be useful, but WITHOUT ANY
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
#  A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#  You should have received a copy of the GNU General Public License along with
#  Zeroth-Meta. If not, see <http://www.gnu.org/licenses/>.
#

"""test_scaler file contains test cases for functions inside scaler.py file in zpmath package."""

__copyright__ = '2023 Zeroth Principles Research'
__license__ = 'GPLv3'
__docformat__ = 'google'
__author__ = 'Zeroth Principles Engineering'
__email__ = 'engineering@zeroth-principles.com'
__authors__ = ['Deepak Singh <deepaksingh@zeroth-principles.com>']

import pandas as pd
import numpy as np
from zpmath.scaler import ZscoreScaler, NormalScaler

def test_NormalScaler():
    # Create a sample dataframe
    df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [6, 7, 8, 9, 10],
        'C': [11, 12, 13, 14, 15]
    })
    axis = 1
    # Create an instance of NormalScaler
    scaler = NormalScaler(dict(axis = axis))

    # Execute the function
    result = scaler(df)

    # Check if the result is a pandas DataFrame
    assert isinstance(result, pd.DataFrame)

    # Check if the result has the same shape as the input dataframe
    assert result.shape == df.shape

    # Check if the result is normalized and transformed to a normal distribution
    assert np.allclose(result.mean(axis=axis), 0)
    assert np.allclose(result.std(axis=axis, ddof=1), 1)

def test_ZscoreScaler():
    # Create a sample dataframe
    df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [6, 7, 8, 9, 10],
        'C': [11, 12, 13, 14, 15]
    })

    # Create an instance of ZscoreScaler
    scaler = ZscoreScaler()
    axis = 1
    ######### test case1
    # recursice method
    result = scaler(df, dict(axis=axis, winsorize_method = "recursive", winsorize_params = dict(max_score = 2.8, eps = 0.3)))

    # Check if the result is a pandas DataFrame
    assert isinstance(result, pd.DataFrame)

    # Check if the result has the same shape as the input dataframe
    assert result.shape == df.shape

    # Check if the result is standardized using z-scores
    assert np.allclose(result.mean(axis = axis), 0)
    assert np.allclose(result.std(axis = axis, ddof = 1), 1)

    ######### test case2
    # scipy method
    result = scaler(df, dict(axis=1, winsorize_method = "scipy", winsorize_params = dict(limits = [0.05, 0.05])))

    # Check if the result is a pandas DataFrame
    assert isinstance(result, pd.DataFrame)

    # Check if the result has the same shape as the input dataframe
    assert result.shape == df.shape

    # Check if the result is standardized using z-scores
    assert np.allclose(result.mean(axis = axis), 0)
    assert np.allclose(result.std(axis = axis, ddof = 1), 1)
