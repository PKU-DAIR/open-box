.. OpenBox documentation master file, created by
   sphinx-quickstart on Fri Dec 23 13:31:47 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:github_url: https://github.com/PKU-DAIR/open-box

###############################################################
OpenBox: Generalized and Efficient Blackbox Optimization System
###############################################################

**OpenBox** is an efficient open-source system designed for **solving
generalized black-box optimization (BBO) problems**, such as
`automatic hyper-parameter tuning <./examples/single_objective_hpo.html>`__,
automatic A/B testing, experimental design, database knob tuning,
processor architecture and circuit design,
resource allocation, automatic chemical design, etc.

The design of **OpenBox** follows the philosophy of providing **"BBO as a service"** - we
opt to implement **OpenBox** as a distributed, fault-tolerant, scalable, and efficient service,
with a wide range of application scope, stable performance across problems
and advantages such as ease of use, portability, and zero maintenance.

There are two ways to use **OpenBox**:
`Standalone python package <./installation/installation_guide.html>`__
and `Online BBO service <./openbox_as_service/service_introduction.html>`__.


------------------------------------------------

Who should consider using OpenBox
---------------------------------

-  Those who want to **tune hyper-parameters** for their ML tasks
   automatically.

-  Those who want to **find the optimal configuration** for their
   configuration search tasks (e.g., database knob tuning).

-  Data platform owners who want to **provide BBO service in their
   platform**.

-  Researchers and data scientists who want to **solve
   generalized BBO problems easily**.

------------------------------------------------

.. _openbox-characteristics--capabilities:

OpenBox capabilities
--------------------------------------

OpenBox has a wide range of functionality scope, which includes:

1. BBO with multiple objectives and constraints.

2. BBO with transfer learning.

3. BBO with distributed parallelization.

4. BBO with multi-fidelity acceleration.

5. BBO with early stops.

In the following, we provide a taxonomy of existing BBO systems:

============== ========== ==== ========== ======= ===========
System/Package Multi-obj. FIOC Constraint History Distributed
============== ========== ==== ========== ======= ===========
Hyperopt       ×          √    ×          ×       √
Spearmint      ×          ×    √          ×       ×
SMAC3          ×          √    ×          ×       ×
BoTorch        √          ×    √          ×       ×
GPflowOPT      √          ×    √          ×       ×
Vizier         ×          √    ×          △       √
HyperMapper    √          √    √          ×       ×
HpBandSter     ×          √    ×          ×       √
**OpenBox**    √          √    √          √       √
============== ========== ==== ========== ======= ===========

-  **FIOC**: Support different input variable types, including
   Float, Integer, Ordinal and Categorical.

-  **Multi-obj.**: Support optimizing multiple objectives.

-  **Constraint**: Support inequality constraints.

-  **History**: Support injecting prior knowledge from previous
   tasks into the current search. (△ means the system cannot support
   it for general cases)

-  **Distributed**: Support parallel evaluations in a distributed
   environment.

------------------------------------------------

Installation
------------

Please refer to our `Installation
Guide <./installation/installation_guide.html>`__.

------------------------------------------------

Quick Start
-----------
In the following, we provide an example of optimizing the Branin function. For more description of this example,
please refer to `Quick Start <./quick_start/quick_start.html>`__.


.. code:: python

    import numpy as np
    from openbox import Optimizer, sp

    # Define Search Space
    space = sp.Space()
    x1 = sp.Real("x1", -5, 10, default_value=0)
    x2 = sp.Real("x2", 0, 15, default_value=0)
    space.add_variables([x1, x2])

    # Define Objective Function
    def branin(config):
       x1, x2 = config['x1'], config['x2']
       y = (x2-5.1/(4*np.pi**2)*x1**2+5/np.pi*x1-6)**2+10*(1-1/(8*np.pi))*np.cos(x1)+10
       return y

    # Run
    if __name__ == '__main__':
       opt = Optimizer(branin, space, max_runs=50, task_id='quick_start')
       history = opt.run()
       print(history)

------------------------------------------------

Documentation
-------------

-  To learn more about OpenBox, refer to `OpenBox
   Overview <./overview/overview.html>`__.

-  To install OpenBox, refer to `OpenBox
   Installation Guide <./installation/installation_guide.html>`__.

-  To get started with OpenBox, refer to `Quick
   Start Tutorial <./quick_start/quick_start.html>`__.


------------------------------------------------

Related Articles
----------------

-  `Tuning LightGBM with
   OpenBox <https://github.com/PKU-DAIR/open-box/blob/master/docs/en/articles/openbox_LightGBM.md>`__

-  `Tuning XGBoost using
   OpenBox <https://github.com/PKU-DAIR/open-box/blob/master/docs/en/articles/openbox_XGBoost.md>`__

------------------------------------------------

Releases and Contributing
-------------------------

OpenBox has a frequent release cycle. Please let us know if you encounter a bug by `filling an
issue <https://github.com/PKU-DAIR/open-box/issues/new/choose>`__.

We appreciate all contributions. If you are planning to contribute any
bug-fixes, please do so without further discussions.

If you plan to contribute new features, new modules, etc. please first
open an issue or reuse an existing issue, and discuss the feature with us.

To learn more about making a contribution to OpenBox, please refer to
our `how-to-contribute page <https://github.com/PKU-DAIR/open-box/blob/master/CONTRIBUTING.md>`__.

We appreciate all contributions and thank all the contributors!

------------------------------------------------

Related Publications
--------------------

**OpenBox: A Generalized Black-box Optimization Service**
Yang Li, Yu Shen, Wentao Zhang, Yuanwei Chen, Huaijun Jiang, Mingchao Liu, Jiawei Jiang, Jinyang Gao, Wentao Wu, Zhi Yang, Ce Zhang, Bin Cui; KDD 2021.

**MFES-HB: Efficient Hyperband with Multi-Fidelity Quality Measurements**
Yang Li, Yu Shen, Jiawei Jiang, Jinyang Gao, Ce Zhang, Bin Cui; AAAI 2021.

**Transfer Learning based Search Space Design for Hyperparameter Tuning.**
Yang Li, Yu Shen, Huaijun Jiang, Tianyi Bai, Wentao Zhang, Ce Zhang, Bin Cui; KDD 2022.

**TransBO: Hyperparameter Optimization via Two-Phase Transfer Learning.**
Yang Li, Yu Shen, Huaijun Jiang, Wentao Zhang, Zhi Yang, Ce Zhang, Bin Cui; KDD 2022.

**PaSca: a Graph Neural Architecture Search System under the Scalable Paradigm.**
Wentao Zhang, Yu Shen, Zheyu Lin, Yang Li, Xiaosen Li, Wen Ouyang, Yangyu Tao, Zhi Yang, and Bin Cui; WWW 2022, 🏆 Best Student Paper Award.

**Hyper-Tune: Towards Efficient Hyper-parameter Tuning at Scale.**
Yang Li, Yu Shen, Huaijun Jiang, Wentao Zhang, Jixiang Li, Ji Liu, Ce Zhang, Bin Cui; VLDB 2022.

--------------------

Related Project
---------------

Targeting at openness and advancing the AutoML ecosystem,
we have also released another open-source project.

-  `MindWare <https://github.com/PKU-DAIR/mindware>`__: an
   open-source system that provides end-to-end ML model training and inference capabilities.

-  `SGL <https://github.com/PKU-DAIR/SGL>`__: a scalable graph learning toolkit for extremely large graph datasets.

------------------------------------------------

Feedback
--------

-  `File an
   issue <https://github.com/PKU-DAIR/open-box/issues>`__ on
   GitHub.

-  Email us via liyang.cs@pku.edu.cn or shenyu@pku.edu.cn.

------------------------------------------------

License
-------

The entire codebase is under `MIT
license <https://github.com/PKU-DAIR/open-box/blob/master/LICENSE>`__.

------------------------------------------------

..  toctree::
    :caption: Table of Contents
    :maxdepth: 2
    :titlesonly:

    Overview <overview/overview>
    Installation <installation/installation_guide>
    Quick Start <quick_start/quick_start>
    Examples <examples/examples>
    Advanced Usage <advanced_usage/advanced_usage>
    OpenBox as Service <openbox_as_service/openbox_as_service>
    Research and Publications <research_and_publications/research_and_publications>
    Change Logs <change_logs/change_logs>

