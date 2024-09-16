import numpy as np
import pandas as pd
import math


def createTestDataFrame(
    number_users: int,
    ratio_baseline: float, 
    conversion_baseline: float, 
    conversion_alternative: float, 
    average_ticket_baseline: float,
    average_ticket_alternative: float,
):

    users_baseline = round(ratio_baseline * number_users)
    paids_baseline = math.ceil(users_baseline * conversion_baseline)
    revenue_total_baseline = paids_baseline * average_ticket_baseline
    shape_baseline = 1 + paids_baseline
    scale_baseline = 1 / (1 + revenue_total_baseline)
    payments_baseline = np.random.gamma(
        shape_baseline, scale_baseline, size=paids_baseline
    )
    payments_baseline = np.array([round(1 / xi, 2) for xi in payments_baseline])
    zeros_baseline = np.repeat(0, users_baseline - paids_baseline)
    result_baseline = np.concatenate((payments_baseline, zeros_baseline))
    baseline = pd.DataFrame({"revenue": result_baseline, "alternative": "baseline"})

    users_alternative = number_users - users_baseline
    paids_alternative = math.ceil(users_alternative * conversion_alternative)
    revenue_total_alternative = paids_alternative * average_ticket_alternative
    shape_alternative = 1 + paids_alternative
    scale_alternative = 1 / (1 + revenue_total_alternative)
    payments_alternative = np.random.gamma(
        shape_alternative, scale_alternative, size=paids_alternative
    )
    payments_alternative = np.array([round(1 / xi, 2) for xi in payments_alternative])
    zeros_alternative = np.repeat(0, users_alternative - paids_alternative)
    result_alternative = np.concatenate((payments_alternative, zeros_alternative))
    alternative = pd.DataFrame({"revenue": result_alternative, "alternative": "test"})

    df = pd.concat([baseline, alternative], ignore_index=True)
    df = df.sample(frac=1)
    df = df.reset_index().rename(columns={"index": "user_id"}).sort_values("user_id")

    return df
