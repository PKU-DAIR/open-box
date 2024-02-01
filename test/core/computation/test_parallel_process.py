import pytest
from openbox.core.computation.parallel_process import ParallelEvaluation
from openbox.core.computation.nondaemonic_processpool import ProcessPool


def objective_func(x):
    return x ** 2


def test_parallel_evaluation_initialization():
    objective_function = objective_func
    with ParallelEvaluation(objective_function) as parallel_evaluation:
        assert parallel_evaluation.n_worker == 1
        assert parallel_evaluation.process_pool is not None
        assert parallel_evaluation.objective_function == objective_function

        param_list = [1, 2, 3]
        results = parallel_evaluation.parallel_execute(param_list)
        assert len(results) == len(param_list)

