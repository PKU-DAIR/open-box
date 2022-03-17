from gpflowopt.bo import *
import time


class BayesianOptimizer_modified(BayesianOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_list = []
        self.global_start_time = time.time()

    def _optimize(self, fx, n_iter):
        """
        Internal optimization function. Receives an ObjectiveWrapper as input. As exclude_gradient is set to true,
        the placeholder created by :meth:`_evaluate_objectives` will not be returned.

        :param fx: :class:`.objective.ObjectiveWrapper` object wrapping expensive black-box objective and constraint functions
        :param n_iter: number of iterations to run
        :return: OptimizeResult object
        """
        assert isinstance(fx, ObjectiveWrapper)

        # Evaluate and add the initial design (if any)
        initial = self.get_initial()
        values = fx(initial)
        self._update_model_data(initial, values)

        # Remove initial design for additional calls to optimize to proceed optimization
        self.set_initial(EmptyDesign(self.domain).generate())

        def inverse_acquisition(x):
            return tuple(map(lambda r: -r, self.acquisition.evaluate_with_gradients(np.atleast_2d(x))))

        # Optimization loop
        for i in range(n_iter):
            # If a callback is specified, and acquisition has the setup flag enabled (indicating an upcoming
            # compilation), run the callback.
            with self.silent():
                if self._model_callback and self.acquisition._needs_setup:
                    self._model_callback([m.wrapped for m in self.acquisition.models])

                result = self.optimizer.optimize(inverse_acquisition)
                self._update_model_data(result.x, fx(result.x))

            self.time_list.append(time.time() - self.global_start_time)  # modified

            if self.verbose:
                metrics = []

                with self.silent():
                    bo_result = self._create_bo_result(True, 'Monitor')
                    metrics += ['MLL [' + ', '.join(
                        '{:.3}'.format(model.compute_log_likelihood()) for model in self.acquisition.models) + ']']

                # fmin
                n_points = bo_result.fun.shape[0]
                if n_points > 0:
                    funs = np.atleast_1d(np.min(bo_result.fun, axis=0))
                    fmin = 'fmin [' + ', '.join('{:.3}'.format(fun) for fun in funs) + ']'
                    if n_points > 1:
                        fmin += ' (size {0})'.format(n_points)

                    metrics += [fmin]

                # constraints
                n_points = bo_result.constraints.shape[0]
                if n_points > 0:
                    constraints = np.atleast_1d(np.min(bo_result.constraints, axis=0))
                    metrics += [
                        'constraints [' + ', '.join('{:.3}'.format(constraint) for constraint in constraints) + ']']

                # error messages
                metrics += [r.message.decode('utf-8') if isinstance(r.message, bytes) else r.message for r in
                            [bo_result, result] if not r.success]

                print('iter #{0:>3} - {1}'.format(
                    i,
                    ' - '.join(metrics)), flush=True)

        return self._create_bo_result(True, "OK")
