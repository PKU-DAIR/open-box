# 复杂搜索空间的问题定义

在快速入门中，我们已经示范了如何定义搜索空间中的独立变量。
然而现实世界中的搜索空间往往是复杂的，可能包含空间内部条件、禁止条款等。
本教程将介绍**Openbox**如何支持复杂的搜索空间。

## 分级条件
目前**Openbox**中的搜索空间是基于包 [ConfigSpace](https://github.com/automl/ConfigSpace) 构建的。
因为 ConfigSpace 本身就支持分级条件，所以我们可以直接使用 ConfigSpace 提供的高级接口。

在下面的例子中我们展示了如何用 `Conditions` 构建一个分级的搜索空间：

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

本例的输出如下，

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

在这个例子中，变量`x2`在`x1`值为'c1'的时候被激活。
变量`x3`在当`x1`值为'c2'或'c3'时被激活。
当变量`x1`的值为'c4'时，`x2`和`x3`都未被激活。
未激活的变量值默认设为'np.nan'。
在优化的过程中，无效的变量组合不会被采样，
因而也就不会需要验证时间。

通过使用 `Conditions`，用户就能构建具有分级结构的复杂搜索空间。
对于 `Conditions` 的更多细节，请参考[ConfigSpace 官方文档](https://automl.github.io/ConfigSpace/main/) 。

## 空间内变量约束
为了支持变量间的约束（例如 `a<b` 或 `a+b<10`）， 我们推荐添加如下例中的采样条件，

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

接口 `set_sample_condition` 需要一个函数，输入一个配置并输出一个布尔值 来指示该配置是否有效。
在优化过程中只有满足采样条件的配置（函数返回值为`True`）能被采样。

注意，与带约束任务中黑盒约束不同的是，空间内变量约束无需运行目标函数就能检查。
