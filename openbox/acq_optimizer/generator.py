import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional, Any,Tuple
from ConfigSpace import Configuration
from ConfigSpace.util import get_one_exchange_neighbourhood
from openbox.utils.history import Observation, History
from openbox.utils.util_funcs import get_types
import warnings
from openbox.acquisition_function.acquisition import AbstractAcquisitionFunction
from openbox.utils.constants import MAXINT
from ..compressor.sampling import SamplingStrategy
import scipy.optimize

MAX_INT = 10000

class SearchGenerator(ABC):    
    @abstractmethod
    def generate(self, 
                 history: History,
                 num_points: int,
                 rng: np.random.RandomState,
                 acq_function=None,
                 **kwargs) -> List[Configuration]:
        pass

class RandomSearchGenerator(SearchGenerator):
    def __init__(self, sampling_strategy:SamplingStrategy=None,config_space=None,random_state=None,batch_size=None):
        if sampling_strategy is not None:
            self.sampling_strategy = sampling_strategy
            self.config_space=sampling_strategy.get_spaces()[0]
        elif config_space is not None:
            self.config_space=config_space
        else:
            raise ValueError("sampling_strategy and config_space is required!") 
        if random_state is None:
            self.random_state='high'
        else:
            self.random_state=random_state

        if batch_size is None:
            types, bounds = get_types(self.config_space)
            dim = np.sum(types == 0)
            self.batch_size = min(5000, max(2000, 200 * dim))
        else:
            self.batch_size = batch_size
        
    def generate(self, 
                 history:History,
                 num_points: int,
                 rng: np.random.RandomState,
                 acq_function=None,
                 **kwargs) -> List[Configuration]:
        if self.random_state=='high':
            configs = self.sampling_strategy.sample(num_points)
            for config in configs:
                config.origin = f'Random Search'
                
        elif self.random_state=='medium':
            from openbox.utils.samplers import SobolSampler
            cur_idx = 0
            configs = list()
            while cur_idx < num_points:
                batch_size = min(self.batch_size, num_points - cur_idx)
                turbo_state = kwargs.get('turbo_state', None)
                if turbo_state is None:
                    lower_bounds = None
                    upper_bounds = None
                else:
                    num_objectives=history.num_objectives
                    if num_objectives > 1:
                        # TODO implement adaptive strategy to choose trust region center for MO
                        raise NotImplementedError()
                    else:
                        incumbent_config = rng.choice(history.get_incumbent_configs())
                        x_center = incumbent_config.get_array()
                        lower_bounds = x_center - turbo_state.length / 2.0
                        upper_bounds = x_center + turbo_state.length / 2.0

                sobol_sampler = SobolSampler(self.config_space, batch_size,
                                         lower_bounds, upper_bounds,
                                         random_state=rng.randint(0, int(1e8)))
                _configs = sobol_sampler.generate(return_config=True)
                configs.extend([_configs[idx] for idx in range(len(_configs))])
                cur_idx += self.batch_size
            for config in configs:
                config.origin = f'BatchMC Search'
                
        elif self.random_state=='low':
            if acq_function is None:
                raise ValueError('acq_function is required!')        
            d=len(self.config_space.get_hyperparameters())
            bound=(0.0,1.0)
            configs=[]
            x_tries = rng.uniform(bound[0], bound[1], size=(num_points, d))
            for i in range(x_tries.shape[0]):
            # convert array to Configuration
                config = Configuration(self.config_space, vector=x_tries[i])
                config.origin = 'MESMO Search'
                configs.append(config)
                
        else:
            raise ValueError('Random_state is invalid!')
        
        return configs

class LocalSearchGenerator(SearchGenerator):    
    def __init__(self, 
                 max_steps: Optional[int] = None,
                 n_steps_plateau_walk: int = 10,
                 remove_duplicates: bool = True,
                 config_space=None,
                 sampling_strategy:SamplingStrategy=None):
        self.max_steps = max_steps
        self.n_steps_plateau_walk = n_steps_plateau_walk#高原行走步数
        self.remove_duplicates = remove_duplicates

        if sampling_strategy is not None:
            self.sampling_strategy=sampling_strategy
            self.config_space=sampling_strategy.get_spaces()[0]
        elif config_space is not None:
            self.config_space=config_space
        else:
            raise ValueError("sampling_strategy and config_space is required!")
    
    def generate(self,
                 history: History,
                 num_points: int,
                 rng: np.random.RandomState,
                 acq_function=None,
                 **kwargs) -> List[Configuration]:
        init_points = self._get_initial_points(
            rng,acq_function, num_points, history)

        configs = []
        # Start N local search from different random start points
        for start_point in init_points:
            acq_val, configuration = self._one_iter(
                rng,acq_function, start_point, self.n_steps_plateau_walk,**kwargs)

            configuration.origin = "Local Search"
            configs.append(configuration)

        if self.remove_duplicates:
            configs=self._remove_duplicates(configs)

        return configs

    def _sort_configs_by_acq_value(
            self,
            rng,
            acquisition_function,
            configs: List[Configuration]
    ) -> List[Tuple[float, Configuration]]:
        """Sort the given configurations by acquisition value

        Parameters
        ----------
        acquisition_function: AbstractAcquisitionFunction
            acquisition function
        configs : list(Configuration)

        Returns
        -------
        list: (acquisition value, Candidate solutions),
                ordered by their acquisition function value
        """

        acq_values = acquisition_function(configs)

        # From here
        # http://stackoverflow.com/questions/20197990/how-to-make-argsort-result-to-be-random-between-equal-values
        random = rng.rand(len(acq_values))
        # Last column is primary sort key!
        indices = np.lexsort((random.flatten(), acq_values.flatten()))

        # Cannot use zip here because the indices array cannot index the
        # rand_configs list, because the second is a pure python list
        return [(acq_values[ind][0], configs[ind]) for ind in indices[::-1]]
    
    def _get_initial_points(self, rng,acquisition_function, num_points, history):

        if history.empty():
            init_points = self.config_space.sample_configuration(
                size=num_points)
        else:
            # initiate local search with best configurations from previous runs
            configs_previous_runs = history.configurations
            configs_previous_runs_sorted = self._sort_configs_by_acq_value(
                rng,acquisition_function, configs_previous_runs)
            num_configs_local_search = int(min(
                len(configs_previous_runs_sorted),
                num_points)
            )
            init_points = list(
                map(lambda x: x[1],
                    configs_previous_runs_sorted[:num_configs_local_search])
            )

        return init_points

    def _one_iter(
            self,
            rng,
            acquisition_function: AbstractAcquisitionFunction, 
            start_point: Configuration,
            n_steps_plateau_walk,
            **kwargs
    ) -> Tuple[float, Configuration]:

        incumbent = start_point
        # Compute the acquisition value of the incumbent
        acq_val_incumbent = acquisition_function([incumbent], **kwargs)[0]
        
        plateau_step=0
        local_search_steps = 0
        neighbors_looked_at = 0
        while True:

            local_search_steps += 1
            changed_inc = False

            # Get one exchange neighborhood returns an iterator (in contrast of
            # the previously returned list).
            all_neighbors = get_one_exchange_neighbourhood(
                incumbent, seed=rng.randint(MAXINT))

            for neighbor in all_neighbors:
                acq_val = acquisition_function([neighbor], **kwargs)
                neighbors_looked_at += 1

                if acq_val > acq_val_incumbent:
                    # logger.debug("Switch to one of the neighbors")
                    incumbent = neighbor
                    acq_val_incumbent = acq_val
                    changed_inc = True
                    plateau_step=0
                    break

            if (self.max_steps is not None and local_search_steps == self.max_steps):
                break
            
            if not changed_inc:
                if plateau_step>=n_steps_plateau_walk:
                    break
                else:
                    plateau_step+=1
            
        return acq_val_incumbent, incumbent
    
    def _remove_duplicates(self, configs: List[Configuration]) -> List[Configuration]:
        seen = set()
        unique = []
        for config in configs:
            key = str(sorted(config.get_dictionary().items()))
            if key not in seen:
                seen.add(key)
                unique.append(config)
        return unique

class CMAESGenerator(SearchGenerator):
    def __init__(self,sampling_strategy:SamplingStrategy=None,config_space=None,sigma=0.5):
        if sampling_strategy is not None:
            self.sampling_strategy=sampling_strategy
            self.config_space=sampling_strategy.get_spaces()[0]
        elif config_space is not None:
            self.config_space=config_space
        else:
            raise ValueError("sampling_strategy and config_space is required!")
        self.sigma=sigma
    
    def generate(self,
                history:History,
                num_points: int,
                rng: np.random.RandomState,
                acq_function=None,
                **kwargs) -> List[Configuration]:
        if acq_function is None:
            raise ValueError("acq_function is required.")
        try:
            from cma import CMAEvolutionStrategy
        except ImportError:
            raise ImportError("Package cma is not installed!")

        types, bounds = get_types(self.config_space)
        assert all(types == 0)

        # Check Constant Hyperparameter
        const_idx = list()
        for i, bound in enumerate(bounds):
            if np.isnan(bound[1]):
                const_idx.append(i)

        hp_num = len(bounds) - len(const_idx)
        es = CMAEvolutionStrategy(hp_num * [0], self.sigma, inopts={'bounds': [0, 1]})
        eval_num = 0
        configs = list()
        while eval_num < num_points:
            X = es.ask(number=es.popsize)
            _X = X.copy()
            for i in range(len(_X)):
                for index in const_idx:
                    _X[i] = np.insert(_X[i], index, 0)
            _X = np.asarray(_X)
            values = acq_function(_X,**kwargs)
            values = np.reshape(values, (-1,))
            es.tell(X, values)
            configs.extend([(values[i], _X[i]) for i in range(es.popsize)])
            eval_num += es.popsize
        configs.sort(reverse=True, key=lambda x: x[0])
        Configs=[_[1] for _ in configs]
        configs=[Configuration(self.config_space,vector=Configs[i],origin = f'CMAES Search') for i in range(num_points)]
        return configs
    
class ScipySearchGenerator(SearchGenerator):
    def __init__(self,sampling_strategy:SamplingStrategy=None,config_space=None,method=None):
        if sampling_strategy is not None:
            self.sampling_strategy = sampling_strategy
            self.config_space=self.sampling_strategy.get_spaces()[0]
        elif config_space is not None:
            self.config_space=config_space
        else:
            raise ValueError("sampling_strategy and config_space is required!")
        self.method=method
    
        
    def generate(self, 
                 history:History,
                 num_points: int,
                 rng: np.random.RandomState,
                 acq_function=None,
                 initial_configs=None,
                 **kwargs) -> List[Configuration]:
        
        if acq_function is None:
            raise ValueError('acq_function is required!')
        
                
        def negative_acq(x):
        # shape of x = (d,)
            x = np.clip(x, 0.0, 1.0)    # fix numerical problem in L-BFGS-B
            try:
                # self.config_space._check_forbidden(x)
                Configuration(self.config_space, vector=x).is_valid_configuration()
            except ValueError:
                return np.inf
            return -acq_function(x,**kwargs)
        
        if self.method=='local':
            types, bounds = get_types(self.config_space)    # todo: support constant hp in scipy optimizer
            assert all(types == 0), 'Scipy optimizer (L-BFGS-B) only supports Integer and Float parameters.'
            self.bounds = bounds
            options = dict(disp=False, maxiter=1000)
            self.scipy_config = dict(tol=None, method='L-BFGS-B', options=options)
            if initial_configs is None:
                if history is None:
                    initial_configs=self.sampling_strategy.sample(num_points)
            
            for init_config in initial_configs:    
                initial_config=init_config.get_array()
        
                configs=list()
                with warnings.catch_warnings():
                # ignore warnings of np.inf
                    warnings.filterwarnings("ignore", message="invalid value encountered in subtract", category=RuntimeWarning)
                    result = scipy.optimize.minimize(fun=negative_acq,
                                                x0=initial_config,
                                                bounds=self.bounds,
                                                **self.scipy_config)   
                try:
                    x = np.clip(result.x, 0.0, 1.0)  # fix numerical problem in L-BFGS-B
                    config = Configuration(self.config_space, vector=x,origin = f'Scipy Search')
                    config.is_valid_configuration()
                    configs.append(config)
                except Exception:
                    pass
                if not configs:
                    raise ValueError()
            
        elif self.method=='global':
            configs = []
            result = scipy.optimize.differential_evolution(func=negative_acq,bounds=self.bounds)
            try:
                x = np.clip(result.x, 0.0, 1.0)  # fix numerical problem in L-BFGS-B
                config = Configuration(self.config_space, vector=x,origin = f'ScipyGlobal Search')
                config.is_valid_configuration()
                configs.append(config)
            except Exception:
                pass
            if not configs:
                raise ValueError()
            
        else:
            raise  ValueError("method should be local or global")
            
        return configs
    