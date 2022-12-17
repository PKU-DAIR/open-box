<p align="center"><a href="https://github.com/PKU-DAIR/open-box">
  <img src="docs/imgs/logo.png" width="40%" alt="OpenBox Logo" >
</a></p>

-----------

[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](
  https://github.com/PKU-DAIR/open-box/blob/master/LICENSE)
[![Issues](https://img.shields.io/github/issues-raw/PKU-DAIR/open-box.svg)](
  https://github.com/PKU-DAIR/open-box/issues?q=is%3Aissue+is%3Aopen)
[![Pull Requests](https://img.shields.io/github/issues-pr-raw/PKU-DAIR/open-box.svg)](
  https://github.com/PKU-DAIR/open-box/pulls?q=is%3Apr+is%3Aopen)
[![Version](https://img.shields.io/github/release/PKU-DAIR/open-box.svg)](
  https://github.com/PKU-DAIR/open-box/releases)
[![Documentation Status](https://readthedocs.org/projects/open-box/badge/?version=latest)](
  https://open-box.readthedocs.io/)

[OpenBox Docs](https://open-box.readthedocs.io) | [OpenBox中文文档](https://open-box.readthedocs.io/zh_CN/latest/)

## OpenBox: Generalized and Efficient Blackbox Optimization System
**OpenBox** is an efficient and generalized blackbox optimization (BBO) system, which supports the following 
characteristics: 1) **BBO with multiple objectives and constraints**, 2) **BBO with transfer learning**, 3) 
**BBO with distributed parallelization**, 4) **BBO with multi-fidelity acceleration** and 5) **BBO with early stops**.
OpenBox is designed and developed by the AutoML team from the [DAIR Lab](http://net.pku.edu.cn/~cuibin/) at Peking 
University, and its goal is to make blackbox optimization easier to apply both in industry and academia, and help 
facilitate data science.


## Software Artifacts
#### Standalone Python package.
Users can install the released package and use it with Python.
#### Distributed BBO service.
We adopt the "BBO as a service" paradigm and implement OpenBox as a managed general service for black-box optimization. 
Users can access this service via REST API conveniently, and do not need to worry about other issues such as environment 
setup, software maintenance, programming, and optimization of the execution. Moreover, we also provide a Web UI,
through which users can easily track and manage the tasks.


## Design Goal

The design of OpenBox follows the following principles:
+ **Ease of use**: Minimal user effort, and user-friendly visualization for tracking and managing BBO tasks.
+ **Consistent performance**: Host state-of-the-art optimization algorithms; Choose the proper algorithm automatically.
+ **Resource-aware management**: Give cost-model-based advice to users, e.g., minimal workers or time-budget.
+ **Scalability**: Scale to dimensions on the number of input variables, objectives, tasks, trials, and parallel 
  evaluations.
+ **High efficiency**: Effective use of parallel resources, system optimization with transfer-learning and 
  multi-fidelities, etc.
+ **Fault tolerance**, **extensibility**, and **data privacy protection**.

## Links
+ [Documentations](https://open-box.readthedocs.io/en/latest/?badge=latest) | 
  [中文文档](https://open-box.readthedocs.io/zh_CN/latest/)
+ [Examples](https://github.com/PKU-DAIR/open-box/tree/master/examples)
+ [Pypi package](https://pypi.org/project/openbox/)
+ Conda package: [to appear soon]()
+ Blog post: [to appear soon]()

## News
+ OpenBox based solutions achieved the First Place of [ACM CIKM 2021 AnalyticCup](https://www.cikm2021.org/analyticup)
  (Track - Automated Hyperparameter Optimization of Recommendation System).
+ OpenBox team won the Top Prize (special prize) in the open-source innovation competition at 
  [2021 CCF ChinaSoft](http://chinasoft.ccf.org.cn/papers/chinasoft.html) conference.
+ [Pasca](https://github.com/PKU-DAIR/SGL), which adopts Openbox to support neural architecture search functionality, 
  won the Best Student Paper Award at WWW'22.

## OpenBox Capabilities in a Glance
<table>
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Build-in Optimization Components</b>
      </td>
      <td>
        <b>Optimization Algorithms</b>
      </td>
      <td>
        <b>Optimization Services</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
      <ul><li><b>Surrogate Model</b></li>
        <ul>
          <li>Gaussian Process</li>
          <li>TPE</li>
          <li>Probabilistic Random Forest</li>
          <li>LightGBM</li>
        </ul>
        </ul>
      <ul>
        <li><b>Acquisition Function</b></li>
          <ul>
           <li>EI</li>
           <li>PI</li>
           <li>UCB</li>
           <li>MES</li>
           <li>EHVI</li>
           <li>TS</li>
          </ul>
      </ul>
        <ul>
        <li><b>Acquisition Optimizer</b></li>
        <ul>
           <li>Random Search</li>
           <li>Local Search</li>
           <li>Interleaved RS and LS</li>
           <li>Differential Evolution</li>
           <li>L-BFGS-B</li>
          </ul>
        </ul>
      </td>
      <td align="left" >
        <ul>
        <li><b>Bayesian Optimization</b></li>
        <ul>
            <li>GP-based BO</li>
            <li>SMAC</li>
            <li>TPE</li>
            <li>LineBO</li>
            <li>SafeOpt</li>
            </ul>
        </ul>
        <ul>
        <li><b>Multi-fidelity Optimization</b></li>
        <ul>
            <li>Hyperband</li>
            <li>BOHB</li>
            <li>MFES-HB</li>
            </ul>
        </ul>
        <ul>
        <li><b>Evolutionary Algorithms</b></li>
        <ul>
            <li>Surrogate-assisted EA</li>
            <li>Regularized EA</li>
            <li>Adaptive EA</li>
            <li>Differential EA</li>
            <li>NSGA-II</li>
            </ul>
        </ul>
        <ul>
        <li><b>Others</b></li>
        <ul>
            <li>Anneal</li>
            <li>PSO</li>
            <li>Random Search</li>
            </ul>
        </ul>
      </td>
      <td>
      <ul>
        <li><a href="https://open-box.readthedocs.io/en/latest/advanced_usage/parallel_evaluation.html">
          Local Machine</a></li>
        <li><a href="https://open-box.readthedocs.io/en/latest/advanced_usage/parallel_evaluation.html">
          Cluster Servers</a></li>
        <li><a href="https://open-box.readthedocs.io/en/latest/advanced_usage/parallel_evaluation.html">
          Hybrid mode</a></li>
        <li><a href="https://open-box.readthedocs.io/en/latest/openbox_as_service/openbox_as_service.html">
          Software as a Service</a></li>
      </ul>
      </td>
    </tr>
  </tbody>
</table>


## Installation

### System Requirements

Installation Requirements:
+ Python >= 3.7 (Python 3.7 is recommended!)

Supported Systems:
+ Linux (Ubuntu, ...)
+ macOS
+ Windows

We **strongly** suggest you to create a Python environment via 
[Anaconda](https://www.anaconda.com/products/individual#Downloads):
```bash
conda create -n openbox python=3.7
conda activate openbox
```

Then we recommend you to update your `pip`, `setuptools` and `wheel` as follows:
```bash
pip install --upgrade pip setuptools wheel
```

### Installation from PyPI

To install OpenBox from PyPI:

```bash
pip install openbox
```

For advanced features, [install SWIG](https://open-box.readthedocs.io/en/latest/installation/install_swig.html)
first and then run `pip install "openbox[extra]"`. 

### Manual Installation from Source

To install the newest OpenBox from the source code, please run the following commands:
```bash
git clone https://github.com/PKU-DAIR/open-box.git && cd open-box
pip install .
```

Also, for advanced features, [install SWIG](https://open-box.readthedocs.io/en/latest/installation/install_swig.html)
first and then run `pip install ".[extra]"`.

For more details about installation instructions, please refer to the 
[Installation Guide Document](https://open-box.readthedocs.io/en/latest/installation/installation_guide.html).

## Quick Start

A quick start example is given by:

```python
import numpy as np
from openbox import Optimizer, space as sp

# Define Search Space
space = sp.Space()
x1 = sp.Real("x1", -5, 10, default_value=0)
x2 = sp.Real("x2", 0, 15, default_value=0)
space.add_variables([x1, x2])

# Define Objective Function
def branin(config):
    x1, x2 = config['x1'], config['x2']
    y = (x2-5.1/(4*np.pi**2)*x1**2+5/np.pi*x1-6)**2+10*(1-1/(8*np.pi))*np.cos(x1)+10
    return {'objectives': [y]}

# Run
if __name__ == '__main__':
    opt = Optimizer(branin, space, max_runs=50, task_id='quick_start')
    history = opt.run()
    print(history)
```

The example with multi-objectives and constraints is as follows:

```python
import matplotlib.pyplot as plt
from openbox import Optimizer, space as sp

# Define Search Space
space = sp.Space()
x1 = sp.Real("x1", 0.1, 10.0)
x2 = sp.Real("x2", 0.0, 5.0)
space.add_variables([x1, x2])

# Define Objective Function
def CONSTR(config):
    x1, x2 = config['x1'], config['x2']
    y1, y2 = x1, (1.0 + x2) / x1
    c1, c2 = 6.0 - 9.0 * x1 - x2, 1.0 - 9.0 * x1 + x2
    return dict(objectives=[y1, y2], constraints=[c1, c2])

# Run
if __name__ == "__main__":
    opt = Optimizer(CONSTR, space, num_objectives=2, num_constraints=2,
                    max_runs=50, ref_point=[10.0, 10.0], task_id='moc')
    history = opt.run()
    history.plot_pareto_front()  # plot for 2 or 3 objectives
    plt.show()
```

We also provide **HTML Visualization** by setting additional options
`visualization`=`basic`/`advanced` and `auto_open_html=True`(optional) in `Optimizer`:

```python
opt = Optimizer(
    ...,
    visualization='advanced',  # or 'basic'. For 'advanced', run 'pip install "openbox[extra]"' first
    auto_open_html=True,       # open the visualization page in your browser automatically
    task_id='example_task',
    logging_dir='logs',
)
history = opt.run()
```

For more visualization details, please refer to 
[HTML Visualization](https://open-box.readthedocs.io/en/latest/visualization/visualization.html).

**More Examples**:
+ [Single-Objective with Constraints](
  https://github.com/PKU-DAIR/open-box/blob/master/examples/optimize_problem_with_constraint.py)
+ [Multi-Objective](https://github.com/PKU-DAIR/open-box/blob/master/examples/optimize_multi_objective.py)
+ [Multi-Objective with Constraints](
  https://github.com/PKU-DAIR/open-box/blob/master/examples/optimize_multi_objective_with_constraint.py)
+ [Ask-and-tell Interface](https://github.com/PKU-DAIR/open-box/blob/master/examples/ask_and_tell_interface.py)
+ [Parallel Evaluation on Local](
  https://github.com/PKU-DAIR/open-box/blob/master/examples/evaluate_async_parallel_optimization.py)
+ [Distributed Evaluation](https://github.com/PKU-DAIR/open-box/blob/master/examples/distributed_optimization.py)
+ [Tuning LightGBM](https://github.com/PKU-DAIR/open-box/blob/master/examples/tuning_lightgbm.py)
+ [Tuning XGBoost](https://github.com/PKU-DAIR/open-box/blob/master/examples/tuning_xgboost.py)

## **Enterprise Users**
<img src="docs/imgs/logo_tencent.png" width="35%" class="align-left" alt="Tencent Logo">

* [Tencent Inc.](https://www.tencent.com/en-us/)

<img src="docs/imgs/logo_alibaba.png" width="35%" class="align-left" alt="Alibaba Logo">

* [Alibaba Group](https://www.alibabagroup.com/en/global/home)

<img src="docs/imgs/logo_kuaishou.png" width="35%" class="align-left" alt="Kuaishou Logo">

* [Kuaishou Technology](https://www.kuaishou.com/en)


## **Releases and Contributing**
OpenBox has a frequent release cycle. Please let us know if you encounter a bug by 
[filling an issue](https://github.com/PKU-DAIR/open-box/issues/new/choose).

We appreciate all contributions. If you are planning to contribute any bug-fixes, 
please do so without further discussions.

If you plan to contribute new features, new modules, etc. please first open an issue or reuse an existing issue, 
and discuss the feature with us.

To learn more about making a contribution to OpenBox, please refer to our 
[How-to contribution page](https://github.com/PKU-DAIR/open-box/blob/master/CONTRIBUTING.md). 

We appreciate all contributions and thank all the contributors!


## **Feedback**
* [File an issue](https://github.com/PKU-DAIR/open-box/issues) on GitHub
* Email us via [*Yang Li*](https://thomas-young-2013.github.io/) or *shenyu@pku.edu.cn*
* [Q&A] Join the QQ group: 227229622

## **Related Projects**

Targeting at openness and advancing AutoML ecosystems, we had also released few other open-source projects.

* [MindWare](https://github.com/PKU-DAIR/mindware): an open source system that provides end-to-end ML model training 
  and inference capabilities.
* [SGL](https://github.com/PKU-DAIR/SGL): a scalable graph learning toolkit for extremely large graph datasets.
* [HyperTune](https://github.com/PKU-DAIR/HyperTune): a large-scale multi-fidelity hyper-parameter tuning system.

## **Related Publications**

**OpenBox: A Generalized Black-box Optimization Service.**
Yang Li, Yu Shen, Wentao Zhang, Yuanwei Chen, Huaijun Jiang, Mingchao Liu, Jiawei Jiang, Jinyang Gao, Wentao Wu,
Zhi Yang, Ce Zhang, Bin Cui; KDD 2021, CCF-A.
https://arxiv.org/abs/2106.00421

**MFES-HB: Efficient Hyperband with Multi-Fidelity Quality Measurements.**
Yang Li, Yu Shen, Jiawei Jiang, Jinyang Gao, Ce Zhang, Bin Cui; AAAI 2021, CCF-A.
https://arxiv.org/abs/2012.03011

**Transfer Learning based Search Space Design for Hyperparameter Tuning.**
Yang Li, Yu Shen, Huaijun Jiang, Tianyi Bai, Wentao Zhang, Ce Zhang, Bin Cui; KDD 2022, CCF-A.
https://arxiv.org/abs/2206.02511

**TransBO: Hyperparameter Optimization via Two-Phase Transfer Learning.**
Yang Li, Yu Shen, Huaijun Jiang, Wentao Zhang, Zhi Yang, Ce Zhang, Bin Cui; KDD 2022, CCF-A.
https://arxiv.org/abs/2206.02663

**PaSca: a Graph Neural Architecture Search System under the Scalable Paradigm.**
Wentao Zhang, Yu Shen, Zheyu Lin, Yang Li, Xiaosen Li, Wen Ouyang, Yangyu Tao, Zhi Yang, and Bin Cui; 
WWW 2022, CCF-A, 🏆 Best Student Paper Award.
https://arxiv.org/abs/2203.00638

**Hyper-Tune: Towards Efficient Hyper-parameter Tuning at Scale.**
Yang Li, Yu Shen, Huaijun Jiang, Wentao Zhang, Jixiang Li, Ji Liu, Ce Zhang, Bin Cui; VLDB 2022, CCF-A.
https://arxiv.org/abs/2201.06834

## **License**

The entire codebase is under [MIT license](LICENSE).
