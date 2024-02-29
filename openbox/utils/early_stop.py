import numpy as np
from openbox import logger


class EarlyStopException(Exception):
    """Exception raised for early stop in Advisor."""
    pass


class EarlyStopAlgorithm(object):
    """
    Class for early stop algorithms.

    Criteria for early stop include:
    1. No Improvement Rounds: Stop if the optimization does not improve over a number of rounds.
    2. Improvement Threshold: Stop if the Expected Improvement is less than a certain percentage of
       the difference between the best objective value and the default objective value.

    Parameters
    ----------
    min_iter : int
        Minimum number of iterations before early stop is considered.
    min_improvement_percentage : float
        The minimum improvement percentage. If the Expected Improvement (EI) is less than
        `min_improvement_percentage * (default_obj_value - best_obj_value)`, early stop is triggered.
        If `improvement_threshold` is 0, this criterion is disabled.
    max_no_improvement_rounds : int
        The maximum tolerable rounds with no improvement before early stop.
        If `max_no_improvement_rounds` is 0, this criterion is disabled.
    """
    def __init__(
            self,
            min_iter: int = 10,
            min_improvement_percentage: float = 0.05,
            max_no_improvement_rounds: int = 10,
    ):
        self.min_iter = min_iter
        self.min_improvement_percentage = min_improvement_percentage
        self.max_no_improvement_rounds = max_no_improvement_rounds
        logger.info(f'Early stop options: '
                    f'min_iter={min_iter}, '
                    f'min_improvement_percentage={min_improvement_percentage}, '
                    f'max_no_improvement_rounds={max_no_improvement_rounds}')

    def check_setup(self, advisor):
        """
        Check if the early stop algorithm is applicable to the given advisor.
        """
        if advisor.num_objectives != 1:
            raise ValueError("Early stop is only supported for single-objective optimization currently.")

        assert self.min_iter > 0, "Minimum number of iterations for early stop must be positive."

        assert self.min_improvement_percentage >= 0, "min_improvement_percentage should be non-negative."
        if self.min_improvement_percentage > 0:
            assert advisor.acq_type == 'ei', ("Using min_improvement_percentage requires the "
                                              "Expected Improvement acquisition function.")

        assert self.max_no_improvement_rounds >= 0, "Maximum number of no improvement rounds should be non-negative."

    @staticmethod
    def already_early_stopped(history):
        return history.meta_info.get('already_early_stopped', False)

    @staticmethod
    def set_already_early_stopped(history):
        history.meta_info['already_early_stopped'] = True

    def decide_early_stop_before_suggest(self, history):
        """
        Determine whether to early stop before suggesting the next configuration.
        """
        if self.already_early_stopped(history):
            logger.info('Early stop already triggered!')
            return True

        if len(history) < self.min_iter:
            return False

        # Condition 1: No improvement over multiple rounds
        if self.max_no_improvement_rounds > 0:
            best_obj = np.inf
            last_improvement_round = 0
            for i, objs in enumerate(history.objectives, start=1):
                if objs[0] < best_obj:
                    best_obj = objs[0]
                    last_improvement_round = i
            no_improvement_rounds = len(history) - last_improvement_round
            if no_improvement_rounds > self.max_no_improvement_rounds:
                logger.info(f'[Early Stop] No improvement over {no_improvement_rounds} rounds!')
                return True

        return False

    def decide_early_stop_after_suggest(self, history, max_acq_value: float = None) -> bool:
        """
        Determine whether to early stop after suggesting the next configuration.
        """
        if len(history) < self.min_iter:
            return False

        # Condition 2: EI less than the threshold
        if self.min_improvement_percentage > 0:
            assert max_acq_value is not None
            default_obj = history.objectives[0][0]  # todo: handle failure
            best_obj = history.get_incumbent_value()
            threshold = self.min_improvement_percentage * (default_obj - best_obj)
            if max_acq_value < threshold:
                logger.info(f'[Early Stop] EI less than the threshold! '
                            f'min_improvement_percentage={self.min_improvement_percentage}, '
                            f'default_obj={default_obj}, best_obj={best_obj}, threshold={threshold}, '
                            f'max_EI={max_acq_value}')
                return True

        return False
