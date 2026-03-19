<p align="center"><a href="https://github.com/PKU-DAIR/open-box">
  <img src="docs/imgs/logo.png" width="40%" alt="OpenBox Logo">
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
[![Test](https://github.com/PKU-DAIR/open-box/actions/workflows/test.yml/badge.svg)](https://github.com/PKU-DAIR/open-box/actions/workflows/test.yml)
[![Documentation Status](https://readthedocs.org/projects/open-box/badge/?version=latest)](
  https://open-box.readthedocs.io/)

[OpenBox Documentation](https://open-box.readthedocs.io)
| [OpenBox中文文档](https://open-box.readthedocs.io/zh_CN/latest/)

## OpenBox: 通用高效的黑盒优化系统
**OpenBox** 是解决黑盒优化（超参数优化）问题的高效且通用的开源系统，支持以下特性： 1) **多目标与带约束的黑盒优化**。2) 
**迁移学习**。3) **分布式并行验证**。4) **多精度优化加速**。5) **早停机制**。6) **搜索空间压缩**。
OpenBox是由北京大学[DAIR实验室](http://net.pku.edu.cn/~cuibin/)自动化机器学习（AutoML）小组设计并开发的，目标是
使黑盒优化在学术界和工业界的应用更加便捷，并促进数据科学的发展。


## 使用方式
#### 本地Python包
用户可以安装我们发布的Python包，从而在本地使用黑盒优化算法。
#### 分布式黑盒优化服务
OpenBox是一个提供通用黑盒优化服务的系统。用户可以使用REST API便捷地访问服务，无需担心环境配置、代码编写与维护、执行优化等问题。
用户还可以通过我们提供的网页用户界面，监控与管理优化任务。


## 设计理念

我们的设计遵循以下理念：
+ **易用**：用户以最小代价使用黑盒优化服务，可通过用户友好的可视化界面监控与管理优化任务。
+ **性能优异**：集成最先进（state-of-the-art）的优化算法，并可自动选择最优策略。
+ **资源感知管理**：为用户提供基于成本（时间、并行数等）的建议。
+ **规模可扩展**：对于输入维度、目标维度、任务数、并行验证数量等有较好的可扩展性。
+ **高效**：充分利用并行资源，并利用迁移学习、多精度优化加速搜索。
+ **错误容忍**、**系统可扩展性**、**数据隐私保护**。

## 链接
+ [Documentations](https://open-box.readthedocs.io/en/latest/) | 
  [中文文档](https://open-box.readthedocs.io/zh_CN/latest/)
+ [样例代码](https://github.com/PKU-DAIR/open-box/tree/master/examples)
+ [Pypi包](https://pypi.org/project/openbox/)
+ Conda包: [即将到来]()
+ 博客: [即将到来]()

<!-- start of news (for docs) -->

## 新闻
+ **OpenBox** based solutions achieved the **First Place** of 
  [ACM CIKM 2021 AnalyticCup](https://www.cikm2021.org/analyticup)
  (Track - Automated Hyperparameter Optimization of Recommendation System).
+ **OpenBox** team won the **Top Prize (special prize)** in the open-source innovation competition at 
  [2021 CCF ChinaSoft](http://chinasoft.ccf.org.cn/papers/chinasoft.html) conference.
+ [**Pasca**](https://github.com/PKU-DAIR/SGL), which adopts Openbox to support neural architecture search 
  functionality, won the **Best Student Paper Award at WWW'22**.

<!-- end of news (for docs) -->

## OpenBox功能概览
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
           <li>CMA-ES</li>
           <li>Upper Bound Maximizer</li>
          </ul>
        </ul>
      <ul>
        <li><b>Search Space Compressor</b></li>
        <ul>
          <li>Dimension Selection (SHAP, Correlation, Adaptive, Expert)</li>
          <li>Range Compression (Boundary, KDE, SHAP-weighted)</li>
          <li>Projection (REMBO, HesBO, KPCA)</li>
          <li>Quantization &amp; Filling</li>
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


## 安装教程

### 系统环境需求

安装需求：
+ Python >= 3.8 （推荐版本为Python 3.8）

支持系统：
+ Linux (Ubuntu, ...)
+ macOS
+ Windows

我们**强烈建议**您为OpenBox创建一个单独的Python环境，例如通过
[Anaconda](https://www.anaconda.com/products/individual#Downloads):
```bash
conda create -n openbox python=3.8
conda activate openbox
```

我们建议您在安装OpenBox之前通过以下命令更新`pip`，`setuptools`和`wheel`：
```bash
pip install --upgrade pip setuptools wheel
```

### 通过PyPI安装（推荐）

使用以下命令通过PyPI安装OpenBox：

```bash
pip install openbox
```

如需使用高级功能，请先[安装 SWIG](https://open-box.readthedocs.io/zh_CN/latest/installation/install_swig.html)
，然后运行 `pip install "openbox[extra]"`。

### 通过源码手动安装

使用以下命令通过Github源码安装OpenBox:
```bash
git clone https://github.com/PKU-DAIR/open-box.git && cd open-box
pip install .
```

同样，如需使用高级功能，请先[安装 SWIG](https://open-box.readthedocs.io/zh_CN/latest/installation/install_swig.html)
，然后运行 `pip install ".[extra]"`。

如果您安装遇到问题，请参考我们的[安装文档](https://open-box.readthedocs.io/zh_CN/latest/installation/installation_guide.html)

## 快速入门

快速入门示例：

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

多目标带约束优化问题示例：

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

我们还提供了**HTML可视化网页**。在`Optimizer`中设置
`visualization`=`basic`/`advanced` 以及 `auto_open_html=True`(可选) 来启用该功能：

```python
opt = Optimizer(...,
    visualization='advanced',  # or 'basic'. For 'advanced', run 'pip install "openbox[extra]"' first
    auto_open_html=True,       # open the visualization page in your browser automatically
)
history = opt.run()
```

对于更多可视化细节，请参考：
[HTML可视化文档](https://open-box.readthedocs.io/zh_CN/latest/visualization/visualization.html)。

**更多示例**：
+ [单目标带约束优化](
  https://github.com/PKU-DAIR/open-box/blob/master/examples/optimize_problem_with_constraint.py)
+ [多目标优化](https://github.com/PKU-DAIR/open-box/blob/master/examples/optimize_multi_objective.py)
+ [多目标带约束优化](
  https://github.com/PKU-DAIR/open-box/blob/master/examples/optimize_multi_objective_with_constraint.py)
+ [Ask-and-tell接口](https://github.com/PKU-DAIR/open-box/blob/master/examples/ask_and_tell_interface.py)
+ [单机并行验证](
  https://github.com/PKU-DAIR/open-box/blob/master/examples/evaluate_async_parallel_optimization.py)
+ [分布式并行验证](https://github.com/PKU-DAIR/open-box/blob/master/examples/distributed_optimization.py)
+ [LightGBM调参](https://github.com/PKU-DAIR/open-box/blob/master/examples/tuning_lightgbm.py)
+ [XGBoost调参](https://github.com/PKU-DAIR/open-box/blob/master/examples/tuning_xgboost.py)

## **企业用户**
<img src="docs/imgs/logo_tencent.png" width="35%" class="align-left" alt="Tencent Logo">

* [腾讯公司](https://www.tencent.com/zh-cn/)

<img src="docs/imgs/logo_alibaba.png" width="35%" class="align-left" alt="Alibaba Logo">

* [阿里巴巴集团](https://www.alibabagroup.com/)

<img src="docs/imgs/logo_kuaishou.png" width="35%" class="align-left" alt="Kuaishou Logo">

* [快手科技](https://www.kuaishou.com/)


## **参与贡献**
如果您在使用OpenBox的过程中遇到Bug，请向我们[提交issue](https://github.com/PKU-DAIR/open-box/issues/new/choose)。
如果您对Bug进行了修复，欢迎直接向我们提交[PR](https://github.com/PKU-DAIR/open-box/pulls)。

如果您想要为OpenBox添加新功能、新模块等，请先开放issue，我们会与您讨论。

如果您想更好地了解如何参与项目贡献，请参考[如何参与贡献](https://github.com/PKU-DAIR/open-box/blob/master/CONTRIBUTING.md)页面。

我们在此感谢所有项目贡献者！


## **反馈**
* 在GitHub上[提交issue](https://github.com/PKU-DAIR/open-box/issues)。
* 通过邮箱联系我们：[*Yang Li*](https://thomas-young-2013.github.io/)，
  *shenyu@pku.edu.cn* 或 *jianghuaijun@pku.edu.cn*
* [Q&A] 加入QQ群：227229622

<!-- start of related projects and publications (for docs) -->

## **相关项目**

以开放性和推进AutoML生态系统为目标，我们还发布了一些其他的开源项目：

* [MindWare](https://github.com/PKU-DAIR/mindware) : 提供端到端机器学习模型训练和预测功能的开源系统。
* [SGL](https://github.com/PKU-DAIR/SGL): 一个用于超大图数据集的可扩展图学习工具包。
* [HyperTune](https://github.com/PKU-DAIR/HyperTune): 大规模多精度超参数调优系统。

## **相关发表文章**

**OpenBox: A Python Toolkit for Generalized Black-box Optimization.**

Huaijun Jiang, Yu Shen, Yang Li, Beicheng Xu, Sixian Du, Wentao Zhang, Ce Zhang, Bin Cui; JMLR 2024, CCF-A.
[[paper]](https://www.jmlr.org/papers/v25/23-0537.html)
[[arxiv]](https://arxiv.org/abs/2304.13339)

**OpenBox: A Generalized Black-box Optimization Service.**

Yang Li, Yu Shen, Wentao Zhang, Yuanwei Chen, Huaijun Jiang, Mingchao Liu, Jiawei Jiang, Jinyang Gao, Wentao Wu,
Zhi Yang, Ce Zhang, Bin Cui; KDD 2021, CCF-A.
[[paper]](https://dl.acm.org/doi/abs/10.1145/3447548.3467061)
[[arxiv]](https://arxiv.org/abs/2106.00421)

**MFES-HB: Efficient Hyperband with Multi-Fidelity Quality Measurements.**

Yang Li, Yu Shen, Jiawei Jiang, Jinyang Gao, Ce Zhang, Bin Cui; AAAI 2021, CCF-A.
[[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/17031)
[[arxiv]](https://arxiv.org/abs/2012.03011)

**Transfer Learning based Search Space Design for Hyperparameter Tuning.**

Yang Li, Yu Shen, Huaijun Jiang, Tianyi Bai, Wentao Zhang, Ce Zhang, Bin Cui; KDD 2022, CCF-A.
[[paper]](https://dl.acm.org/doi/abs/10.1145/3534678.3539369)
[[arxiv]](https://arxiv.org/abs/2206.02511)

**TransBO: Hyperparameter Optimization via Two-Phase Transfer Learning.**

Yang Li, Yu Shen, Huaijun Jiang, Wentao Zhang, Zhi Yang, Ce Zhang, Bin Cui; KDD 2022, CCF-A.
[[paper]](https://dl.acm.org/doi/abs/10.1145/3534678.3539255)
[[arxiv]](https://arxiv.org/abs/2206.02663)

**PaSca: a Graph Neural Architecture Search System under the Scalable Paradigm.**

Wentao Zhang, Yu Shen, Zheyu Lin, Yang Li, Xiaosen Li, Wen Ouyang, Yangyu Tao, Zhi Yang, and Bin Cui; 
WWW 2022, CCF-A, 🏆 Best Student Paper Award.
[[paper]](https://dl.acm.org/doi/abs/10.1145/3485447.3511986)
[[arxiv]](https://arxiv.org/abs/2203.00638)

**Hyper-Tune: Towards Efficient Hyper-parameter Tuning at Scale.**

Yang Li, Yu Shen, Huaijun Jiang, Wentao Zhang, Jixiang Li, Ji Liu, Ce Zhang, Bin Cui; VLDB 2022, CCF-A.
[[paper]](https://dl.acm.org/doi/abs/10.14778/3514061.3514071)
[[arxiv]](https://arxiv.org/abs/2201.06834)


## **引用**

如果您使用 OpenBox，请考虑引用以下文章：


```
@inproceedings{li2021openbox,
  title={Openbox: A generalized black-box optimization service},
  author={Li, Yang and Shen, Yu and Zhang, Wentao and Chen, Yuanwei and Jiang, Huaijun and Liu, Mingchao and Jiang, Jiawei and Gao, Jinyang and Wu, Wentao and Yang, Zhi and others},
  booktitle={Proceedings of the 27th ACM SIGKDD conference on knowledge discovery \& data mining},
  pages={3209--3219},
  year={2021}
}

@article{JMLR:v25:23-0537,
  author  = {Huaijun Jiang and Yu Shen and Yang Li and Beicheng Xu and Sixian Du and Wentao Zhang and Ce Zhang and Bin Cui},
  title   = {OpenBox: A Python Toolkit for Generalized Black-box Optimization},
  journal = {Journal of Machine Learning Research},
  year    = {2024},
  volume  = {25},
  number  = {120},
  pages   = {1--11},
  url     = {http://jmlr.org/papers/v25/23-0537.html}
}
```

<!-- end of related projects and publications (for docs) -->

## **许可协议**

我们的代码遵循[MIT许可协议](LICENSE)。
