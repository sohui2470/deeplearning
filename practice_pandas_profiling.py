# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import pandas as pd
import pandas_profiling
data = pd.read_csv('c:\ipynb\spam.csv', encoding='latin1')

data[:5]

pr = data.profile_report()

pr.to_file('.pr_report.html')

pr


