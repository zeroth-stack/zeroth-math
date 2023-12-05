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

"""Superclasses for frequently used design patterns."""

__copyright__ = '2023 Zeroth Principles Research'
__license__ = 'GPLv3'
__docformat__ = 'google'
__author__ = 'Zeroth Principles Engineering'
__email__ = 'engineering@zeroth-principles.com'
__authors__ = ['Deepak Singh <deepaksingh@zeroth-principles.com>']
__all__ = ["winsorize"]

import pandas as pd
from zpmeta.funcs.func import Func
import numpy as np
from scipy.stats import zscore

class RecursiveWinsorize(Func):
    """This function takes a pandas series and winsorize the data by replacing values that are z standard deviations far from the mean with the value at
      z standard deviations.

    """
    def _std_params(self, name: str = None) -> dict:
        """
        Standard parameters for the function class.
        return: dict
            max_score : float
                Threshold for winsorizing the data, here score represents number of standard deviation away.
            
            eps : float
                eps is the margin of error that is accepted around max score to get the solution avoid recurssion error.

        """
        return dict(max_score = 2.8, eps = 0.3)
    @staticmethod
    def check_consistency(operand=None, params: dict = None):
        if not isinstance(operand, pd.Series):
            return ValueError("Only series are allowed")
        
    def _execute(self, operand: pd.Series =None, params: dict = None) -> object:
        prev_upper_bound, prev_lower_bound = None, None
        count=0
        if operand.count()>1:
            while count<50000:
                mean_x = operand.mean()
                sd_x = operand.std()
                upper_bound = mean_x + params["max_score"]*sd_x
                lower_bound = mean_x - params["max_score"]*sd_x
                zscore_max = ((operand - mean_x)/sd_x).abs().max()
                if (prev_lower_bound==lower_bound and prev_upper_bound==upper_bound) or zscore_max<= params["max_score"]+params["eps"]:
                    return operand
                operand[operand>upper_bound] = upper_bound
                operand[operand<lower_bound] = lower_bound
                prev_lower_bound=lower_bound  
                prev_upper_bound=upper_bound
                count+=1
            
            raise ValueError("Winsorization didn't converge after 50000 iterations")
        else:
            if abs(operand.sum()) >  params["max_score"] + params["eps"]:
                operand.loc[~operand.isna()]= params["max_score"]+params["eps"]
            
            return operand
            


