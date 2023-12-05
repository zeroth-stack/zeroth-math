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
__all__ = ["NormalScaler", "ZscoreScaler", "GaussRankScaler"]
import pandas as pd
from zpmeta.funcs.func import Func
from scipy.stats import norm, zscore
from scipy.stats.mstats import winsorize as scipy_winsorize
from zpmath.winsorize import RecursiveWinsorize
import numpy as np

class NormalScaler(Func):
    """This function takes a pandas dataframe and returns a normalized dataframe after ranking the data and transforming to normal distribution.

    """
    def _std_params(self, name: str = None) -> dict:
        """
        Standard parameters for the function class.
        return: dict
            axis : {0 or 'index', 1 or 'columns'}, default 1

        """
        return dict(axis=1)
    
    def _execute(self, operand: pd.DataFrame =None, params: dict = None) -> object:
        print(params)
        rank = operand.rank(axis=params["axis"])
        uniform_dist = rank.div(rank.count(axis=params["axis"])+1, axis= abs(1 - params["axis"]))
        zscore_array = zscore(norm.ppf(uniform_dist), axis = params["axis"], nan_policy = "omit", ddof = 1)
        gauss_rank = pd.DataFrame(data = zscore_array, index = uniform_dist.index, columns = uniform_dist.columns)
        return gauss_rank

class ZscoreScaler(Func):
    """This function takes a pandas dataframe and returns a zscore of the data based on the params"""

    def _std_params(self, name: str = None) -> dict:
        """
        Standard parameters for the function class.
        return: dict
            axis : {0 or 'index', 1 or 'columns'}, default 1

            winsorize_method : str
                "recursive": recursive winsorization method, uses recursize methodology
                "scipy": scipy winsorization method, uses scipy.stats.mstats.winsorize

            winsorize_params : dict
                Below params are for recursive method
                    max_score : float
                    Threshold for winsorizing the data, here score represents number of standard deviation away.
                
                    eps : float
                        eps is the margin of error that is accepted around max score to get the solution avoid recurssion error.

                Below params are for scipy method
                    Look at the documentation of scipy.stats.mstats.winsorize

        """
        return dict(axis = 1, winsorize_method = "recursive", winsorize_params = dict(max_score = 2.8, eps = 0.3))
    
    def _execute(self, operand: pd.DataFrame =None, params: dict = None) -> object:
        if params["winsorize_method"] == "recursive":
            winsorized = operand.apply(lambda x: RecursiveWinsorize()(x, params= params["winsorize_params"]), axis = params["axis"])
        elif params["winsorize_method"] == "scipy":
            # series is required for scipy winsorize otherwise it will return a numpy array
            winsorized = operand.apply(lambda x: pd.Series(scipy_winsorize(x, limits = params["winsorize_params"]["limits"], nan_policy="omit"), index = x.index), 
                                       axis = params["axis"])
        else:
            winsorized = operand.copy()
        zscored = zscore(winsorized, axis = params["axis"], nan_policy = "omit", ddof =1)
        return zscored


