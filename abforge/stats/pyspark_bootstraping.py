from typing import List, Union
import random as rd
from pyspark.sql import SparkSession, DataFrame as SparkDataFrame


def get_bootstraped_mean(data: List[Union[float, int]]) -> float:
    n_samples = len(data)
    samples = [data[rd.randint(0, n_samples - 1)] for _ in range(0, n_samples)]
    return sum(samples) / n_samples


def get_parallel_bootstrap(
    function,
    data: List[Union[float, int]],
    num_samples: int,
    spark_session: SparkSession,
) -> SparkDataFrame:
    rdd = spark_session.sparkContext.parallelize(list(range(1, num_samples + 1)))
    df = (
        rdd.map(lambda x: (x, function(data)))
        .toDF()
        .withColumnRenamed("_1", "sample")
        .withColumnRenamed("_2", "sample_metric")
    )
    return df
