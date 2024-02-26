.. OpenBox documentation master file, created by
   sphinx-quickstart on Fri Dec 23 13:31:47 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:github_url: https://github.com/PKU-DAIR/open-box

###############################
OpenBox: 高效通用的黑盒优化系统
###############################

**OpenBox** 是一个高效的开源系统，旨在解决泛化的黑盒优化（BBO）问题，
例如 :ref:`自动化超参数调优 <examples/single_objective_hpo:单目标的黑盒优化>` 、自动化A/B测试、
实验设计、数据库参数调优、处理器体系结构和电路设计、资源分配、自动化学设计等。

**OnenBox** 的设计理念是将BBO作为一种服务提供给用户。
我们的目标是将 **OpenBox** 实现为一个分布式的、有容错、可扩展的、高效的服务。
它能够对各种应用场景提供广泛的支持，并保证稳定的性能。
**OpenBox** 简单易上手、方便移植和维护。


您可以使用以下两种方法使用 **OpenBox**：
:ref:`单独的Python包 <installation/installation_guide:安装指南>`
和 :ref:`在线BBO服务 <openbox_as_service/service_introduction:服务简介>`。

**OpenBox GitHub:** https://github.com/PKU-DAIR/open-box

------------------------------------------------

.. include:: ../../README.md
   :parser: myst_parser.sphinx_
   :start-after: <!-- start of news (for docs) -->
   :end-before: <!-- end of news (for docs) -->

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

============== ====== ========== ========== ======= ===========
系统/包         FIOC   多目标      约束条件     历史    分布式
============== ====== ========== ========== ======= ===========
Hyperopt       √      ×          ×          ×       √
Spearmint      ×      ×          √          ×       ×
SMAC3          √      √          ×          ×       ×
BoTorch        ×      √          √          ×       ×
Ax             √      √          √          ×       √
Optuna         √      √          √          ×       √
GPflowOPT      ×      √          √          ×       ×
Vizier         √      ×          △          △       √
HyperMapper    √      √          √          ×       ×
HpBandSter     √      ×          ×          ×       √
**OpenBox**    √      √          √          √       √
============== ====== ========== ========== ======= ===========

-  **FIOC**: 支持不同的输入变量类型，包括 Float, Integer, Ordinal 和 Categorical。

-  **多目标**: 支持多目标优化。

-  **约束条件**: 支持不等式约束条件。

-  **历史**: 支持将以前任务的先验知识融入到当前搜索中（即迁移学习）。

-  **分布式**: 支持在分布式环境中并行评估。

-  △ 表示系统在通用场景下不支持或需要安装额外的依赖。

------------------------------------------------


..  toctree::
    :caption: 目录
    :maxdepth: 2
    :titlesonly:

    概览 <overview/overview>
    安装 <installation/installation_guide>
    快速入门 <quick_start/quick_start>
    使用实例 <examples/examples>
    可视化 (New!) <visualization/visualization>
    高级用法 <advanced_usage/advanced_usage>
    OpenBox服务 <openbox_as_service/openbox_as_service>
    Developer's Guide <developers_guide/developers_guide>
    Maintainer's Guide <maintainers_guide/maintainers_guide>
    研究成果 <research_and_publications/research_and_publications>
    更新历史 <change_logs/change_logs>
