#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 22:19:35 2020

@author: nova
"""

import sklearn.datasets as skd
import pandas as pd
import numpy as np
shape_host = (16,3000)
cluster_host = (4,4)
shape_sym = (16,2000)
cluster_sym = (4,4)

host, row_host, col_host = skd.make_checkerboard(shape_host,
                             cluster_host,
                             noise = 10.0,
                             shuffle = False,
                             random_state = 2020)

sym, row_sym, col_sym = skd.make_checkerboard(shape_sym,
                             cluster_sym,
                             noise = 10.0,
                             shuffle = False,
                             random_state = 212)


host_ = pd.DataFrame(np.round(host))
host_ = host_.sample(n = 3000,replace = False, random_state = 12,axis = 1)
sym_ = pd.DataFrame(np.round(sym)).sample(n = 2000,replace = False, random_state = 321,axis = 1)

host_.to_csv("host.csv",index = False)
sym_.to_csv("algae.csv",index = False)