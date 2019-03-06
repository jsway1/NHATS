# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 1.0.2
#   kernel_info:
#     name: python3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Relationship between IFS and the percent automatable
#
# The following code shows the relatiionship between log$_{10}$(IFS) and the percent automatable, using graphs and linear regression. 
#
# It is evident from initial graphs that this relationship is hard to ascertain when IFS is kept in the natural linear scale,
# so we will log-transform it. 
#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

d = pd.read_excel('~/Downloads/Injury Data 03-05-2019.xlsx')
d.columns = d.columns.str.lower()
d.columns = d.columns.str.replace(' ', '_')

d['logIFS'] = np.log10(d['injury_frequency_and_severity'])

# Note that we don't lose any data when doing this transformation, since all the values of `injury_frequency_and_severity` are strictly positive.

np.sum(d['injury_frequency_and_severity'] <= 0)

# # Plotting

sns.jointplot(x = 'percent_automatable', y = 'logIFS', data = d, kind = 'reg', line_kws={'color':'red'});

# # Modeling

X = d['percent_automatable']
X = sm.add_constant(X)
results = sm.OLS(d['logIFS'], X).fit()
print(results.summary())

# In the above model, we find that `percent_automatable` has a coefficient of 0.007. Since we have modeled IFS on the log$_{10}$ scale, this now has a multiplicative interpretation. This model indicates that for every 1% increase in automatability, the IFS increases by a factor of $10^{0.007} = 1.016$, or 1.6%. So, for example, if automatability increases from 20% to 25%, we expect IFS to increase by a factor of $10^{0.007 \times 5} = 1.084$ or 8.4%.

results.pvalues

# The slope coefficient has a p-value of the order of 10$^{-9}$, which indicates high statistical significance. 

# ## Alternative coding using formulas

results2 = smf.ols('np.log10(injury_frequency_and_severity) ~ percent_automatable', data = d).fit()
print(results2.summary())

# ## Residual analysis

sns.residplot(x = 'percent_automatable', y = 'logIFS', data = d, lowess=True, line_kws={'color':'red', 'linewidth':5});

# This looks decent, with no obvious strong pattern in the residuals
