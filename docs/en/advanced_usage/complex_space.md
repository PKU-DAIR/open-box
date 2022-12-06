# Problem Definition with Complex Search Space

In quick examples, we have shown how to define independent variables in the search space.
However, real-world search spaces are often complex, which may contain in-space conditions, forbidden clauses, etc.
In this tutorial, we will show how OpenBox supports complex search spaces.

## Hierarchical Conditions
The search space in OpenBox is currently built on the package [ConfigSpace](https://github.com/automl/ConfigSpace).
While ConfigSpace supports hierarchical conditions, we can directly use the advanced APIs provided in ConfigSpace.

In the following example, we provide an example of building a hierarchical search space using `Conditions`:

```python
from openbox import space as sp
from ConfigSpace import EqualsCondition, InCondition

space = sp.Space()
x1 = sp.Categorical("x1", choices=["c1", "c2", "c3", "c4"])
x2 = sp.Real("x2", -5, 10, default_value=0)
x3 = sp.Real("x3", 0, 15, default_value=0)

equal_condition = EqualsCondition(x2, x1, "c1")  # x2 is active when x1 = c1
in_condition = InCondition(x3, x1, ["c2", "c3"])  # x3 is active when x1 = c2 or x1 = c3

space.add_variables([x1, x2, x3])
space.add_conditions([equal_condition, in_condition])

print(space.sample_configuration(5))
```

The example output can be as follows,

```
[Configuration(values={
  'x1': 'c4',
})
, Configuration(values={
  'x1': 'c1',
  'x2': -4.246561408224157,
})
, Configuration(values={
  'x1': 'c3',
  'x3': 1.7213163807467695,
})
, Configuration(values={
  'x1': 'c3',
  'x3': 13.8469881579991,
})
, Configuration(values={
  'x1': 'c2',
  'x3': 2.9833423891692763,
})
]
```

In this example, the variable `x2` is active when the value of `x1` is c1.
The variable `x3` is active when the value of `x1` is c2 or c3.
When the value of `x1` is c4, neither `x2` nor `x3` is active.
The value of inactive variables are set to np.nan by default.
During optimization, invalid variable combinations will not be sampled,
so there will be no evaluation time spent for invalid configurations.

By utilizing `Conditions`, users can build complex search space with hierarchical structure.
For more details about `Conditions`, please refer to the [ConfigSpace documentation](https://automl.github.io/ConfigSpace/main/) for more details.

## In-space variable Constraints
To support constraints between variables (e.g., `a<b` or `a+b<10`), we recommend adding a sampling condition as shown in the following example,

```python
from openbox import space as sp

def sample_condition(config):
    # require x1 <= x2 and x1 * x2 < 100
    if config['x1'] > config['x2']:
        return False
    if config['x1'] * config['x2'] >= 100:
        return False
    return True
    # return config['x1'] <= config['x2'] and config['x1'] * config['x2'] < 100

cs = sp.ConditionedSpace()
cs.add_variables([...])
cs.set_sample_condition(sample_condition)  # set the sample condition after all variables are added

```

The API `set_sample_condition` requires a function that takes a configuration as the input and output a boolean value,
which indicates whether the configuration is valid.
Only configurations that meet the sample condition (with a return value of `True`) are sampled during optimization.

Note that, the in-space variable constraints are constraints that can be checked directly without running the objective function,
which are different from the black-box constraints in constrained problems.
