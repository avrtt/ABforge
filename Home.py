import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats
from scipy.stats import norm

import abforge
from abforge.stats.experiment import Experiment, Variant

from utils import save_results_in_session_state

import logging

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="ABforge: all-in-one Bayesian A/B testing", page_icon="")

st.markdown(
    """
# All-in-one Bayesian A/B testing

**[ABforge](https://github.com/avrtt/ABforge)** is a tool to automate your decisions in A/B test experiments with two alternatives, where 
you want to know which one is better. It's focused on conversion and revenue evaluation, typical in e-commerce, and uses Bayesian statistics to achieve faster and more insightful results. 

No sample size is obligatory, since you can get partial results and make decisions, but feel free to estimate it to have an idea.

The engine is more powerful than common online websites because it measures statistics for 3 variables at once: 
- conversion rate
- value for conversions (e.g. revenue, cost, time spent on page, etc.) 
- average value per impression (e.g. Average Revenue per User, Cost Per User, etc.)

"""
)

st.markdown("""## Implemented pages""")

st.markdown(
    """
There are 3 pages in web UI so far, each one comes with default examples to understand:

- **Analyze with summary information**: most simple use case, you have the summary data for the whole test and just want results;
- **Analyze with summary CSV**: you have a CSV with summary information per day/week/etc and want results;
- **Analyze with per impression CSV**: you have a CSV with results per impression and want results. This is the best approach of all if data is available, since we have a better understanding of how sales and revenue is really distributed along the data.
"""
)
