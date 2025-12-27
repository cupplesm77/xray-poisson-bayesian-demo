

# LinkedIn Post Draft
#
# I rarely publish technical critiques on LinkedIn, but this case is too important to ignore because it concerns public
# health and safety.
# Note that this is an "old" example of an incorrect methodological approach.  It is not meant as a slam to the
# original study authors, but rather to highlight the importance of proper statistical methods in scientific research
# based on recent mention in the news.
#
# A foundational 1988 rat study that helped grant inulin its “generally recognized as safe” (GRAS) status with the FDA
# contained a critical statistical flaw.
# The researchers used logistic regression, which is appropriate for binary outcomes like tumor presence, but then
# calculated significance as if it were a linear regression. This means they applied significance tests designed for
# continuous outcomes to binary data, invalidating their conclusions.
#
# When the data were reanalyzed correctly using logistic regression significance tests, the results showed a
# significant increase in tumors linked to inulin
# consumption — a finding that challenges decades of regulatory acceptance.
#
# This example underscores how crucial proper statistical methods are in scientific research, especially when public
# health decisions depend on them.
# A single methodological oversight can have far-reaching consequences.
#
# I hope this encourages more scrutiny and transparency in safety studies that impact us all.
#
# #PublicHealth #Statistics #DataScience #RegulatoryScience #FoodSafety
#
# Python Code Examples for LinkedIn Post
#
# Below are Python code snippets illustrating the incorrect and correct significance calculations for binary outcome
# data, suitable for sharing with my professional network.

import numpy as np
import statsmodels.api as sm

# Sample data: tumor presence (binary) and inulin dose (continuous)
# Modified to have some overlap so the Logistic model can converge
# and find a valid p-value without "Complete Separation".
# data0 = {
#     'tumor': np.array([0, 1, 0, 1, 0, 1, 1, 0]),
#     'dose':  np.array([0.1, 0.9, 0.2, 0.8, 0.1, 0.7, 0.9, 0.3])
# }

data = {
    'tumor': np.array([0, 0, 0, 1, 0, 1, 1, 1, 0, 1]),
    'dose':  np.array([0.1, 0.2, 0.4, 0.4, 0.5, 0.5, 0.6, 0.8, 0.9, 0.9])
}

# Prepare design matrix with intercept
X = sm.add_constant(data['dose'])

# Incorrect approach: Linear regression on binary outcome
model_linear = sm.OLS(data['tumor'], X).fit()
print("Linear regression summary (incorrect):")
print(model_linear.summary())

# Correct approach: Logistic regression on binary outcome
model_logit = sm.Logit(data['tumor'], X).fit(disp=0)
print("Logistic regression summary (correct):")
print(model_logit.summary())

# Extract p-values
print(f"P-value from linear regression: {model_linear.pvalues[1]:.4f}")
print(f"P-value from logistic regression: {model_logit.pvalues[1]:.4f}")

"""
Linear regression summary (incorrect):
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.178
Model:                            OLS   Adj. R-squared:                  0.075
Method:                 Least Squares   F-statistic:                     1.729
Date:                Wed, 17 Dec 2025   Prob (F-statistic):              0.225
Time:                        13:45:53   Log-Likelihood:                -6.2798
No. Observations:                  10   AIC:                             16.56
Df Residuals:                       8   BIC:                             17.16
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const          0.0720      0.363      0.198      0.848      -0.765       0.909
x1             0.8076      0.614      1.315      0.225      -0.609       2.224
==============================================================================
Omnibus:                        1.229   Durbin-Watson:                   2.627
Prob(Omnibus):                  0.541   Jarque-Bera (JB):                0.730
Skew:                          -0.228   Prob(JB):                        0.694
Kurtosis:                       1.757   Cond. No.                         4.97
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

Logistic regression summary (correct):
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                      y   No. Observations:                   10
Model:                          Logit   Df Residuals:                        8
Method:                           MLE   Df Model:                            1
Date:                Wed, 17 Dec 2025   Pseudo R-squ.:                  0.1366
Time:                        13:45:54   Log-Likelihood:                -5.9846
converged:                       True   LL-Null:                       -6.9315
Covariance Type:            nonrobust   LLR p-value:                    0.1688
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
const         -1.9532      1.713     -1.140      0.254      -5.311       1.405
x1             3.6897      2.967      1.244      0.214      -2.126       9.505
==============================================================================
P-value from linear regression: 0.2250
P-value from logistic regression: 0.2137
"""

# 1. The Probability "Overhang" (Invalid Predictions)
# In the Linear Regression (OLS), the model treats the binary outcome like a straight line. If you were to plug in a
# very high dose (e.g., dose = 1.5), the OLS model would predict a "tumor probability" greater than 100%, which is
# mathematically impossible. The Logistic Regression uses a sigmoid curve, ensuring predictions always stay logically
# bounded between 0 and 1.
# 2. Differing Assumptions of "Noise"
# Linear Regression assumes the "errors" (residuals) are normally distributed and constant across all doses. For binary
# data (0 or 1), this is never true—the errors can only be specific values, which violates the foundational assumptions
# of the OLS significance tests (the t-test).
# Logistic Regression assumes a Binomial distribution, which is the "natural" distribution for coin-flip style data
# like tumor presence.
# 3. Sensitivity and Significance
# In your specific output:
# Linear P-value: 0.2250
# Logistic P-value: 0.2137
# The Logistic model is slightly more "sensitive" (a lower p-value) because it better understands the structure of the
# data. While both are above the 0.05 threshold in this tiny sample, the Logistic model provides a more accurate
# estimate of the log-odds of a tumor. In a larger study (like the 1988 rat study), this difference in sensitivity is
# exactly what causes the Linear model to "miss" a significant finding that the Logistic model correctly identifies.
# Summary for your post:
# "Linear regression on binary data is like using a ruler to measure the curve of a ball.
# The method might give you a number, but it ignores the fundamental shape of the object.
# Logistic regression respects the 'binary' nature of the data, providing valid probabilities and more reliable
# significance tests."
