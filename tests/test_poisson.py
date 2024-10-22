import pytest

from abforge.experiments import PoissonDataTest


@pytest.fixture
def conv_test():
    cv = PoissonDataTest()
    cv.add_variant_data("A", [4, 3, 4, 3, 5, 2, 7, 2, 4, 4])
    cv.add_variant_data("B", [3, 8, 4, 5, 4, 2, 2, 3, 4, 6])
    cv.add_variant_data_agg("C", 11, 4, 39, a_prior=2, b_prior=2)
    cv.add_variant_data_agg("D", 10, 4.5, 42)
    cv.add_variant_data_agg("D", 10, 4.5, 42, replace=False)
    cv.add_variant_data_agg("D", 10, 4.5, 42, replace=True)
    cv.delete_variant("D")
    return cv


@pytest.fixture
def conv_test_assessment():
    cv = PoissonDataTest()
    cv.add_variant_data("A", [4, 3, 4, 3, 5, 2, 7, 2, 4, 4])
    cv.add_variant_data("B", [3, 8, 4, 5, 4, 2, 2, 3, 4, 6])
    return cv


def test_variants(conv_test):
    assert conv_test.variant_names == ["A", "B", "C"]


def test_totals(conv_test):
    assert conv_test.totals == [10, 10, 11]


def test_sums(conv_test):
    assert conv_test.sums == [38, 41, 39]


def test_obs_means(conv_test):
    assert conv_test.obs_means == [3.8, 4.1, 4]


def test_means(conv_test):
    assert conv_test.means == [3.54545, 3.81818, 3.53846]


def test_stdevs(conv_test):
    assert conv_test.stdevs == [0.56773, 0.58916, 0.52172]


def test_bounds(conv_test):
    assert conv_test.bounds == [[2.52116, 4.74163], [2.75181, 5.05647], [2.5906, 4.63181]]


def test_a_priors(conv_test):
    assert conv_test.a_priors == [1, 1, 2]


def test_b_priors(conv_test):
    assert conv_test.b_priors == [1, 1, 2]


def test_probabs_of_being_best(conv_test):
    pbbs = conv_test._probabs_of_being_best(sim_count=2000000, seed=314)
    assert pbbs == {"A": 0.266836, "B": 0.480775, "C": 0.252389}


def test_expected_loss(conv_test):
    loss = conv_test._expected_loss(sim_count=2000000, seed=314)
    assert loss == {"A": 0.5896207, "B": 0.3169076, "C": 0.5965555}


def test_expected_loss_prop(conv_test):
    conv_test.evaluate(sim_count=2000000, seed=314)
    loss = conv_test.exp_loss
    assert loss == [0.5896207, 0.3169076, 0.5965555]


def test_probabs_of_being_best_prop(conv_test):
    conv_test.evaluate(sim_count=2000000, seed=314)
    pbbs = conv_test.chance_to_beat
    assert pbbs == [0.266836, 0.480775, 0.252389]


def test_uplift(conv_test):
    conv_test.evaluate(sim_count=2000000, seed=314)
    uplift = conv_test.uplift_vs_a
    assert uplift == [0, 0.07692, -0.00197]


@pytest.mark.mpl_image_compare
def test_poisson_plot(conv_test):
    conv_test.evaluate(sim_count=2000000, seed=314)
    fig = conv_test.plot_distributions(control="A")
    return fig


def test_evaluate_assessment(conv_test_assessment):
    eval_report, cf_pbbs, assessment = conv_test_assessment.evaluate(
        control="A", rope=0.5, closed_form=True, sim_count=2000000, seed=314
    )

    assert (
        eval_report
        == [
            {
                "bounds": [2.52116, 4.74163],
                "expected_loss": 0.4793632,
                "mean": 3.54545,
                "obs_mean": 3.8,
                "prob_being_best": 0.3683965,
                "total": 10,
                "uplift_vs_a": 0,
                "variant": "A",
            },
            {
                "bounds": [2.75181, 5.05647],
                "expected_loss": 0.2066501,
                "mean": 3.81818,
                "obs_mean": 4.1,
                "prob_being_best": 0.6316035,
                "total": 10,
                "uplift_vs_a": 0.07692,
                "variant": "B",
            },
        ]
        and cf_pbbs == [0.36878, 0.63122]
        and assessment
        == {
            "confidence": "Low",
            "decision": "If you were to stop testing now, you could select either " "variant.",
            "lower_bound": -1.33075,
            "upper_bound": 1.8851,
        }
        != {
            "confidence": "Low",
            "decision": "Stop and implement either variant.",
            "lower_bound": -1.33075,
            "upper_bound": 1.8851,
        }
    )


def test_evaluate(conv_test):
    eval_report, cf_pbbs, _ = conv_test.evaluate(closed_form=True, sim_count=2000000, seed=314)
    assert eval_report == [
        {
            "bounds": [2.52116, 4.74163],
            "expected_loss": 0.5896207,
            "mean": 3.54545,
            "obs_mean": 3.8,
            "prob_being_best": 0.266836,
            "total": 10,
            "uplift_vs_a": 0,
            "variant": "A",
        },
        {
            "bounds": [2.75181, 5.05647],
            "expected_loss": 0.3169076,
            "mean": 3.81818,
            "obs_mean": 4.1,
            "prob_being_best": 0.480775,
            "total": 10,
            "uplift_vs_a": 0.07692,
            "variant": "B",
        },
        {
            "bounds": [2.5906, 4.63181],
            "expected_loss": 0.5965555,
            "mean": 3.53846,
            "obs_mean": 4,
            "prob_being_best": 0.252389,
            "total": 11,
            "uplift_vs_a": -0.00197,
            "variant": "C",
        },
    ] and cf_pbbs == [0.32411, 0.5672, 0.10868]
