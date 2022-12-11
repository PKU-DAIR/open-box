.. OpenBox documentation master file, created by
   sphinx-quickstart on Fri Dec 23 13:31:47 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:github_url: https://github.com/PKU-DAIR/open-box

###############################
OpenBox: 高效通用的黑盒优化系统
###############################

**OpenBox** 是一个高效的开源系统，旨在解决泛化的黑盒优化（BBO）问题，
例如 `自动化超参数调优 <./examples/single_objective_hpo.html>`__ 、自动化A/B测试、
实验设计、数据库参数调优、处理器体系结构和电路设计、资源分配、自动化学设计等。

**OnenBox** 的设计理念是将BBO作为一种服务提供给用户。
我们的目标是将 **OpenBox** 实现为一个分布式的、有容错、可扩展的、高效的服务。
它能够对各种应用场景提供广泛的支持，并保证稳定的性能。
**OpenBox** 简单易上手、方便移植和维护。


您可以使用以下两种方法使用 **OpenBox**：
`单独的Python包 <./installation/installation_guide.html>`__
和 `在线BBO服务 <./openbox_as_service/service_introduction.html>`__ 。


------------------------------------------------

OpenBox 针对的用户群体
---------------------------------

-  想要为ML任务自动执行 **超参数调优** 的用户。

-  想要为配置搜索任务找到 **最佳配置** 的用户（例如，数据库参数调优）。

-  想要为数据平台提供 **BBO服务** 的用户。

-  想要方便地解决 **通用BBO问题** 的研究员和数据科学家。

------------------------------------------------

.. _openbox-characteristics--capabilities:

OpenBox 的功能特性
--------------------------------------

OpenBox 有很多强大的功能和特性，包括：

1、 提供多目标和带约束条件的 BBO 服务支持。

2、 提供带迁移学习的 BBO 服务。

3、 提供分布式并行的 BBO 服务。

4、 提供多精度加速的 BBO 服务。

5、 提供带提前终止的 BBO 服务。

下表给出了现有BBO系统的分类：

============== ========== ====== ========== ======= ===========
系统/包         多目标      FIOC   约束条件    历史    分布式
============== ========== ====== ========== ======= ===========
Hyperopt       ×          √      ×          ×       √
Spearmint      ×          ×      √          ×       ×
SMAC3          ×          √      ×          ×       ×
BoTorch        √          ×      √          ×       ×
GPflowOPT      √          ×      √          ×       ×
Vizier         ×          √      ×          △       √
HyperMapper    √          √      √          ×       ×
HpBandSter     ×          √      ×          ×       √
**OpenBox**    √          √      √          √       √
============== ========== ====== ========== ======= ===========

-  **FIOC**: 支持不同的输入变量类型，包括 Float, Integer, Ordinal 和 Categorical。

-  **多目标**: 支持多目标优化。

-  **约束条件**: 支持不等式约束条件。

-  **历史**: 支持将以前任务的先验知识融入到当前搜索中。（ △ 表示系统在通用场景下不支持）

-  **分布式**: 支持在分布式环境中并行评估。

------------------------------------------------

安装
------------

请参考我们的 `安装指南 <./installation/installation_guide.html>`__.

------------------------------------------------

快速上手
-----------
下面我们给出一个优化 Branin 函数的简单实例。更多的细节描述请参考我们的 `快速上手指南 <./quick_start/quick_start.html>`__ 。


.. code:: python

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
       return y

    # Run
    if __name__ == '__main__':
       opt = Optimizer(branin, space, max_runs=50, task_id='quick_start')
       history = opt.run()
       print(history)

------------------------------------------------

文档
-------------

-  想要进一步了解OpenBox？ 请参考 `OpenBox概览 <./overview/overview.html>`__ 。

-  想要安装OpenBox？ 请参考 `OpenBox安装指南 <./installation/installation_guide.html>`__ 。

-  想要快速上手OpenBox？ 请参考 `快速上手指南 <./quick_start/quick_start.html>`__ 。


------------------------------------------------

相关文章
----------------

-  `使用OpenBox对LightGBM进行超参数优化 <https://github.com/PKU-DAIR/open-box/blob/master/docs/zh_CN/articles/openbox_LightGBM.md>`__

-  `使用OpenBox对XGBoost进行超参数优化 <https://github.com/PKU-DAIR/open-box/blob/master/docs/zh_CN/articles/openbox_XGBoost.md>`__

------------------------------------------------

版本发布和贡献
-------------------------

OpenBox 有着频繁的更新周期。
如果你在使用过程中遇到了bug，请在Github上告知我们：`如何提交 issue <https://github.com/PKU-DAIR/open-box/issues/new/choose>`__ 。

我们感谢所有的贡献。如果您需要修复任何bug，请直接修复，无需通知我们。

如果您想要为OpenBox添加新的特性和模块，请先开启一个issue或复用一个现有的issue，然后和我们进一步讨论。



想要了解更多关于为OpenBox提供贡献的方法，请参考我们的 `如何 contribute <https://github.com/PKU-DAIR/open-box/blob/master/CONTRIBUTING.md>`__ 。

我们再次感谢所有为我们提供宝贵建议的开发者！

------------------------------------------------

研究成果
--------------------

**OpenBox: A Generalized Black-box Optimization Service**
Yang Li, Yu Shen, Wentao Zhang, Yuanwei Chen, Huaijun Jiang, Mingchao Liu, Jiawei Jiang, Jinyang Gao, Wentao Wu, Zhi Yang, Ce Zhang, Bin Cui; ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD 2021).

**MFES-HB: Efficient Hyperband with Multi-Fidelity Quality Measurements**
Yang Li, Yu Shen, Jiawei Jiang, Jinyang Gao, Ce Zhang, Bin Cui; The Thirty-Fifth AAAI Conference on Artificial Intelligence (AAAI 2021).

**Transfer Learning based Search Space Design for Hyperparameter Tuning.**
Yang Li, Yu Shen, Huaijun Jiang, Tianyi Bai, Wentao Zhang, Ce Zhang, Bin Cui; KDD 2022.

**TransBO: Hyperparameter Optimization via Two-Phase Transfer Learning.**
Yang Li, Yu Shen, Huaijun Jiang, Wentao Zhang, Zhi Yang, Ce Zhang, Bin Cui; KDD 2022.

**PaSca: a Graph Neural Architecture Search System under the Scalable Paradigm.**
Wentao Zhang, Yu Shen, Zheyu Lin, Yang Li, Xiaosen Li, Wen Ouyang, Yangyu Tao, Zhi Yang, and Bin Cui; WWW 2022, 🏆 Best Student Paper Award.

**Hyper-Tune: Towards Efficient Hyper-parameter Tuning at Scale.**
Yang Li, Yu Shen, Huaijun Jiang, Wentao Zhang, Jixiang Li, Ji Liu, Ce Zhang, Bin Cui; VLDB 2022.

--------------------

相关项目
---------------

以开放性为目标，为了推进AutoML生态系统的发展，我们还发布了一些其他开源项目：

-  `MindWare <https://github.com/PKU-DAIR/mindware>`__ ：一个开源的，提供自动化且端到端的ML模型训练和预测的系统。

-  `SGL <https://github.com/PKU-DAIR/SGL>`__: 一个适用于极大型图数据集的可扩展图学习工具箱

------------------------------------------------

反馈
--------

-  `提交 issue <https://github.com/PKU-DAIR/open-box/issues>`__
-  Email：liyang.cs@pku.edu.cn 或 shenyu@pku.edu.cn

------------------------------------------------

许可证
-------
OpenBox项目基于 `MIT License <https://github.com/PKU-DAIR/open-box/blob/master/LICENSE>`__

------------------------------------------------


..  toctree::
    :caption: 目录
    :maxdepth: 2
    :titlesonly:

    概览 <overview/overview>
    安装 <installation/installation_guide>
    快速入门 <quick_start/quick_start>
    使用实例 <examples/examples>
    可视化 <visualization/visualization>
    高级用法 <advanced_usage/advanced_usage>
    OpenBox服务 <openbox_as_service/openbox_as_service>
    研究成果 <research_and_publications/research_and_publications>
    更新历史 <change_logs/change_logs>
