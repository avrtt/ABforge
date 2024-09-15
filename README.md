**ABforge** is an A/B testing library designed for a variety of use cases, utilizing the Bayesian approach to implement different variables and tests, including binary, Poisson, normal, delta-lognormal and discrete tests on different metrics. It's especially useful for analysing key metrics in marketplaces, such as conversion rates, ticket size, ARPU differences between test variants, etc.

The main idea of this library is to create an all-in-one tool for running A/B tests separately from closed-source code and business logic.

## Installation

Clone & navigate:
```
git clone git@github.com:avrtt/ABforge.git
cd ABforge
```

Using a virtual environment is optional, but recommended. If you'd like to create one, run:
```
python3 -m venv venv
```

Then activate the virtual environment:
- On Linux/macOS:
    ```
    source venv/bin/activate
    ```
- On Windows:
    ```
    venv\Scripts\activate
    ```

    ⚠️ **This project hasn't been tested on Windows.**
     <br><br>

Executing the Makefile will install the required dependencies and the library itself:
```
make
```

<br>

Once installed, you can now import the `abforge` library in your Python code with a simple `import abforge`.

## Usage

Besides importing directly, you can also interact with some parts of the library engine through a Streamlit-based web interface. To run it, simply execute:
```
streamlit run Home.py
```

Below you can discover all released methods of the library.

## Description

This engine measures statistics for 3 very important variables at once: 
- **conversion rate** (e.g., percentage of visits that turn into sales)
- **monetary value for conversions** (e.g., revenue per transaction) 
- **average value per impression** (e.g., Average Revenue per User)

Sometimes, conversion rate isn't the best metric for your test when the most important is if you're bringing more money to the table. That's why ARPU helps you a lot. Revenue also helps you to undestand how your ticket sale is affected between variants.

In frequentist approach:
1. p-value is difficult to understand and has no business value;
2. we can't make an informative conclusion from insignificant test;
3. test isn't valid without fixing the sample size;
4. bigger sample size is required.

Instead, in Bayesian approach:
1. results have clear business value and are easy to understand;
2. we always get valid results and can make at least an informed decision;
3. fixing the sample size isn't required;
4. smaller sample size is sufficient.

### Tests
WIP

### Metrics
WIP

### Decision rules
WIP

### Closed form solutions
WIP

### Error tolerance
WIP

## To do
- [ ] Documentation page
- [ ] Unit tests (methods)
- [x] Integration tests (**Linux**)
- [ ] Integration tests (**Windows**)
- [x] Python 3.10 > 3.12 migration (upd: fix numpy==1.26.4)
- [ ] Remove **setup.py**
- [ ] Add usage example(s)
- [x] Build Streamlit app
- [x] Add references
- [ ] Add images
- [ ] Add logo (Streamlit page & README)

## References
- [Wikipedia: A/B testing](https://en.wikipedia.org/wiki/A/B_testing)
- [Bayesian A/B testing at VWO](https://vwo.com/downloads/VWO_SmartStats_technical_whitepaper.pdf)
- [Optional stopping in data collection: p values, Bayes factors, credible intervals, precision](
  http://doingbayesiandataanalysis.blogspot.com/2013/11/optional-stopping-in-data-collection-p.html)
- [Its time to rethink A/B Testing](https://www.gamedeveloper.com/business/it-s-time-to-re-think-a-b-testing)
- [Agile A/B testing with Bayesian Statistics and Python](https://web.archive.org/web/20150419163005/http://www.bayesianwitch.com/blog/2014/bayesian_ab_test.html)
- [Probabalistic programming and Bayesian methods for hackers](https://nbviewer.org/github/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/tree/master/)
- [Continuous Monitoring of A/B Tests without Pain: Optional Stopping in Bayesian Testing](https://arxiv.org/pdf/1602.05549.pdf)
- [Think Bayes 2](https://allendowney.github.io/ThinkBayes2/index.html)
- [Conjugate prior](https://en.wikipedia.org/wiki/Conjugate_prior)
- [Binomial distributions](https://www.youtube.com/watch?v=8idr1WZ1A7Q)
- [Bayesian Inference 2019](https://vioshyvo.github.io/Bayesian_inference/index.html)
- [An Introduction to Bayesian Thinking](https://statswithr.github.io/book/)
- [Formulas for Bayesian A/B Testing](https://www.evanmiller.org/bayesian-ab-testing.html)
- [Easy Evaluation of Decision Rules in Bayesian A/B testing](https://www.chrisstucchio.com/blog/2014/bayesian_ab_decision_rule.html)
- [Bayesian Data Analysis, Third Edition](http://www.stat.columbia.edu/~gelman/book/BDA3.pdf)
- [Bayes theorem, the geometry of changing beliefs](https://www.youtube.com/watch?v=HZGCoVF3YvM)
- [The quick proof of Bayes' theorem](https://www.youtube.com/watch?v=U_85TaXbeIo)
- [Is Bayesian A/B Testing Immune to Peeking? Not Exactly](http://varianceexplained.org/r/bayesian-ab-testing/)


## Contributing
Feel free to open PRs and issues.

## License
Distributed under the Apache 2.0 License. See LICENSE.txt for more information.

