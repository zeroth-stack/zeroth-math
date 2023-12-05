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

"""test_winsorize file contains test cases for functions inside winsorize.py file in zpmath package."""

__copyright__ = '2023 Zeroth Principles Research'
__license__ = 'GPLv3'
__docformat__ = 'google'
__author__ = 'Zeroth Principles Engineering'
__email__ = 'engineering@zeroth-principles.com'
__authors__ = ['Deepak Singh <deepaksingh@zeroth-principles.com>']

import pandas as pd
import numpy as np
from zpmath.winsorize import RecursiveWinsorize
from scipy.stats import zscore

def test_RecursiveWinsorize():
    # Create a sample pandas series
    series = pd.Series([1, 2, 3, 4, 5])

    # Create an instance of RecursiveWinsorize
    winsorizer = RecursiveWinsorize(dict(max_score = 2.8, eps = 0.3))

    # Execute the function
    result = winsorizer(series)

    # Check if the result is a pandas Series
    assert isinstance(result, pd.Series)

    # Check if the result has the same length as the input series
    assert len(result) == len(series)

    # Check if the result is winsorized within the specified threshold
    assert zscore(result, ddof  = 1, nan_policy="omit").max()<= 3.1
    assert zscore(result, ddof  = 1, nan_policy="omit").min()>= -3.1

