.. _autograd-mechanics:

自动梯度方法（Autograd）: PyTorch的一个RST实例
===============================================

本文将概述autograd的工作原理并记录相关操作。
用户不必完全理解本文，但我们建议您熟悉它，因为它将帮助您编写更高效、更整洁的程序，并可以帮助您进行调试。


.. _excluding-subgraphs:

倒序移除子图
---------------------------------

每个 Tensor 都有一个标记: :attr:`requires_grad` 。它允许我们在进行梯度计算时能进行细粒度的子图移除，从而提升性能。

.. _excluding-requires_grad:

``requires_grad``
^^^^^^^^^^^^^^^^^


如果对于一个操作有一个单独的输入需要梯度，它的输出就需要梯度。
相反，只有所有的输入都不需要梯度时，输出才不需要提出。
倒序计算不会在子图中进行，这里所有的张量都不需要梯度。

.. code::

    >>> x = torch.randn(5, 5)  # requires_grad=False by default
    >>> y = torch.randn(5, 5)  # requires_grad=False by default
    >>> z = torch.randn((5, 5), requires_grad=True)
    >>> a = x + y
    >>> a.requires_grad
    False
    >>> b = a + z
    >>> b.requires_grad
    True

当你想部分冻结你的模型时，或者你事先知道你不打算使用渐变w.r.t.一些参数时，以上内容是特别有用的。
例如，如果你想要细粒度调优一个预训练的CNN时，在冻结基准上转换:attr:`requires_grad`标志即可。
直到计算到达最后一层，中间缓冲区才会被保存。
这时仿射变换将使用需要梯度的权值，网络的输出也将需要它们。


.. code::

    model = torchvision.models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    # Replace the last fully-connected layer
    # Parameters of newly constructed modules have requires_grad=True by default
    model.fc = nn.Linear(512, 100)

    # Optimize only the classifier
    optimizer = optim.SGD(model.fc.parameters(), lr=1e-2, momentum=0.9)

.. _how-autograd-encodes-history:

自动梯度（autograd）如何编码历史
--------------------------------

Autograd是一种反向自动求微分的系统。
从概念上讲，autograd记录了一个图，它记录了在执行操作时创建数据的所有操作，给出了一个有向无环图，其叶是输入张量，根是输出张量。
通过从根到叶跟踪此图，我们可以使用链式规则自动计算梯度。


在内部，autograd将这个图表示为一个由 :class:`Function` 对象（真实的表达式）组成的图。
这个通过:meth:`~torch.autograd.Function.apply` 来对这个图计算评估结果。
当计算前向传播时，同时执行请求的计算，并构建一个表示计算梯度的函数图
（每个 :class:`torch.Tensor` 的 ``.grad_fn`` 特性都是这个图的一个输入点）。
当向前传播的过程完成后，我们进行反向传播计算梯度。


需要注意的一点是，这个图在每次迭代时都是从头开始重新创建的。
这也是我们允许使用任意Python控制流语句的原因，这些语句可以在每次迭代时更改图的整体形状和大小。
在开始训练之前，您不必对所有可能的路径进行编码。

使用autograd的就地操作(In-place operation)
---------------------------------

支持autograd中的就地操作是一件困难的事情，在大多数情况下，我们不鼓励使用它们。
激进的的缓冲区释放和重用策略会使得Autograd非常有效。
只有在很少有情况下，就地操作才能够显著降低内存使用量。
除非您在巨大的内存压力下工作，否则您可能永远都不需要使用它们。

我们不推荐使用就地操作的原因有以下两种：

1. 就地操作可能会覆盖一些计算梯度时所需的值。

2. 就地操作需要重写计算图。
在正常情况下，我们只需分配新对象并保留对旧图的引用，而就地操作则需要将所有输入的创建者更改为 :class:`Function` 来表示此操作。
这可能很棘手，特别是如果有许多张量都引用同一个存储（例如，通过索引或转置创建的存储），并且如果修改后的输入的存储被任何其他 :class:`Tensor` 引用，则就地操作功能就会出错。



就地操作的正确性检验
^^^^^^^^^^^^^^^^^^^^^^^^^^^

每个张量都有一个版本计数器，每次它在任何操作中被标记为dirty时，该计数器都会递增。
当函数为反向传播保存任何张量时，也会保存其包含的张量的版本计数器。
一旦访问``self.saved_tensors`` 时，它就会被选中，如果它大于保存的值，则会引发错误。
这样做可以确保，如果我们使用的是内建函数，并且目前没有检测到任何错误，则可以确保计算梯度的结果是正确的。


多线程 Autograd
----------------------

Autograd引擎负责运行所有的计算后向传播所需的后向操作。
本节将描述所有可以帮助您在多线程环境中充分利用它的细节（这只与PyTorch 1.6+相关，因为以前版本中的行为有所不同）。

用户可以使用多线程代码（例如Hogwild training）来训练他们的模型，并且不阻塞并发的反向计算，示例代码可以是：

.. code::

    # Define a train function to be used in different threads
    def train_fn():
        x = torch.ones(5, 5, requires_grad=True)
        # forward
        y = (x + 3) * (x + 4) * 0.5
        # backward
        y.sum().backward()
        # potential optimizer update


    # User write their own threading code to drive the train_fn
    threads = []
    for _ in range(10):
        p = threading.Thread(target=train_fn, args=())
        p.start()
        threads.append(p)

    for p in threads:
        p.join()


注意到用户需要注意以下行为：

CPU 并发
^^^^^^^^^^^^^^^^^^

当你通过CPU的多线程，使用Python或C++ API运行``backward()`` 或 ``grad()``时，
您期望看到额外的并发性，而不是在执行期间以特定的顺序串行化所有的反向调用（PyTorch 1.6之前的行为）。


非确定性
^^^^^^^^^^^^^^^

如果您在多个线程上并发调用``backward()`` 但使用共享输入（即Hogwild CPU训练）。
由于参数是跨线程自动共享的，因此在跨线程的向后调用上，梯度累加可能变得不确定，因为两个向后调用可能访问并尝试累加相同的 ``.grad`` 属性。
这在技术上是不安全的，它可能会导致的竞争条件和无效使用。

但如果您使用多线程方法来驱动整个训练过程，但使用共享参数，那这个结果就是意料之中的。
使用多线程的用户应该记住线程模型，并且应该预期会发生这种情况。
用户可以使用函数API :func:`torch.autograd.grad` 来计算梯度，而不是 ``backward()``，从而避免不确定性。

图保持
^^^^^^^^^^^^^^^

如果自动梯度图的一部分在线程之间共享，即，单线程运行前向传播的第一部分，而后以多线程的方式运行第二部分，
那么这个图的第一部分就是被共享的。
在这种情况下，在同样的图上执行 ``grad()`` 或 ``backward()`` 的不同线程可能会在一个线程运行时破坏该图。
在这种情况下，另一个线程也会崩溃。
Autograd会向用户发出类似于调用 ``backward()`` 两次的错误信息，但不包含 ``retain_graph=True``，
并且让用户知道他们应该使用 ``retain_graph=True``。


Autograd结点上的线程安全
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

由于Autograd允许调用者线程驱动其反向传播以实现潜在的并行性，因此我们必须确保CPU上共享部分/全部GraphTask的反向传播过程的线程安全。

因为GIL，自定义Python的 ``autograd.function`` 是自动线程安全的。
对于内建的 C++ Autograd结点（例如，累计梯度、复制切片）和自定义的 ``autograd::Function``，
自动梯度引擎使用线程互斥锁来保证自动梯度结点的线程安全，这些结点可能会有状态读写。


C++ hooks 上的非线程安全
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

AutoGad依赖于用户编写的线程安全的C++钩子。
如果您希望钩子正确应用于多线程环境，您需要编写适当的线程锁定代码，以确保钩子是线程安全的。

.. _complex_autograd-doc:

复域上的自动梯度
----------------------------

简而言之：

- 当你使用PyTorch对任何定义在复域或共域上的函数 :math:`f(z)` 求导时，梯度是在假设了部分很大的实值损失函数 :math:`g(input)=L` 时被计算出来的。
  梯度计算公式是 :math:`\frac{\partial L}{\partial z^*}` （注意 z 的共轭），它的负值正好是梯度下降算法中最陡下降的方向。
  因此，所有现有的优化器都可以使用复参数进行现成的操作。
- 这个约定和TensorFlow中计算复微分的约定匹配，但和 JAX 不同（它计算
  :math:`\frac{\partial L}{\partial z}`）。
- 如果您有一个内部使用复操作的实数到实数的函数，这里的约定并不重要：您将始终得到与仅使用实操作实现时相同的结果。

如果你对数学细节感兴趣，或者想知道如何在PyTorch中定义复导数，请继续阅读。

什么是复导数
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

复可微性的数学定义取导数的极限定义，并将其推广到复数上。考虑一个函数 :math:`f: ℂ → ℂ`,

    .. math::
        `f(z=x+yj) = u(x, y) + v(x, y)j`

这里 :math:`u` 和 :math:`v` 是两个变量的实函数。

根据求导的定义，我们有：

    .. math::
        f'(z) = \lim_{h \to 0, h \in C} \frac{f(z+h) - f(z)}{h}

为了满足这个限制, :math:`u` 和 :math:`v` 必须是实可导的, 而且 :math:`f` 也必须满足 `Cauchy-Riemann 等式
<https://en.wikipedia.org/wiki/Cauchy%E2%80%93Riemann_equations>`_.
换言之：为实部和虚部计算的极限（:math:`h`）必须相等。这是一个更为严格的条件。

复可微函数通常称为全纯函数。
它们表现良好，具有从实可微函数中所能看到的所有优良特性，但在优化领域中却几乎没有用处。
对于优化问题，由于复数不是有序域的一部分，所有研究界只使用实值目标函数，因此复数的损失函数没有多大意义。

结果还表明，没有有趣的实值目标函数满足Cauchy-Riemann方程。
所以同态函数的理论不能用于优化，因此大多数人使用Wirtinger演算。

Wirtinger 微积分 ...
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

所以，我们有复可微性和全纯函数的伟大理论，但却无法使用，因为许多常用的函数都不是全纯函数。
那么我们应该怎么做？
Wirtinger 观察到，纵使 :math:`f(z)` 不 holomorphic, 我们也可以把它重写成两个变量函数：
:math:`f(z, z*)` 他们总是 holomorphic的。
这是因为:math:`z` 的实部和虚部可以被表示成 :math:`z` 和 :math:`z^*`：

    .. math::
        \begin{aligned}
            Re(z) &= \frac {z + z^*}{2} \\
            Im(z) &= \frac {z - z^*}{2j}
        \end{aligned}

Wirtinger 微积分提出研究 :math:`f(z, z^*)` ，它在 :math:`f` 实可导时是 holomorphic 的。（另一种理解是坐标系的变化，从 :math:`f(x, y)`
到 :math:`f(z, z^*)`）。
这个函数有偏导
:math:`\frac{\partial }{\partial z}` 和 :math:`\frac{\partial}{\partial z^{*}}`。
我们可以使用链式规则在这些偏导和偏导 w.r.t. 之间建立一个关系 :math:`z`：


    .. math::
        \begin{aligned}
            \frac{\partial }{\partial x} &= \frac{\partial z}{\partial x} * \frac{\partial }{\partial z} + \frac{\partial z^*}{\partial x} * \frac{\partial }{\partial z^*} \\
                                         &= \frac{\partial }{\partial z} + \frac{\partial }{\partial z^*}   \\
            \\
            \frac{\partial }{\partial y} &= \frac{\partial z}{\partial y} * \frac{\partial }{\partial z} + \frac{\partial z^*}{\partial y} * \frac{\partial }{\partial z^*} \\
                                         &= 1j * (\frac{\partial }{\partial z} - \frac{\partial }{\partial z^*})
        \end{aligned}

从上式中，我们有：

    .. math::
        \begin{aligned}
            \frac{\partial }{\partial z} &= 1/2 * (\frac{\partial }{\partial x} - 1j * \frac{\partial }{\partial y})   \\
            \frac{\partial }{\partial z^*} &= 1/2 * (\frac{\partial }{\partial x} + 1j * \frac{\partial }{\partial y})
        \end{aligned}

这是Wirtinger积分的经典定义。

这个变化有很多优美的性质，例如：

- Cauchy-Riemann方程可以被简单地解释为 :math:`\frac{\partial f}{\partial z^*} = 0` （即，函数 :math:`f` 可以被完全用 :math:`z` 表达，而无需 :math:`z^*`）
- 另一个重要的结果我们即将用到，它指出当我们优化一个实值损失函数时，我们在变量更新时应该采取如下的步骤：:math:`\frac{\partial Loss}{\partial z^*}` （而不是 :math:`\frac{\partial Loss}{\partial z}`）。


更多资料请参考 https://arxiv.org/pdf/0906.4835.pdf


Wirtinger微积分在优化中如何被使用？
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在音频和其他领域的研究人员通常使用梯度下降法来优化具有复变量的实值损失函数。
通常，这些人将实值和虚值视为可以更新的独立通道。

对于步长 :math:`s/2` 和损失函数
:math:`L`，我们可以在 :math:`ℝ^2` 上写出下列等式：

    .. math::
        \begin{aligned}
            x_{n+1} &= x_n - (s/2) * \frac{\partial L}{\partial x}  \\
            y_{n+1} &= y_n - (s/2) * \frac{\partial L}{\partial y}
        \end{aligned}

上面等式是如何变换到复数域 :math:`ℂ`上的呢？

    .. math::
        \begin{aligned}
            z_{n+1} &= x_n - (s/2) * \frac{\partial L}{\partial x} + 1j * (y_n - (s/2) * \frac{\partial L}{\partial y}) \\
                    &= z_n - s * 1/2 * (\frac{\partial L}{\partial x} + j \frac{\partial L}{\partial y}) \\
                    &= z_n - s * \frac{\partial L}{\partial z^*}
        \end{aligned}

有趣的事情发生了：
Wirtinger 积分告诉我们，我们可以将上述复变量更新的公式简化为只涉及共轭Wirtinger导数的形式。
由于共轭Wirtinger导数给了我们实值损失函数正确的一步，
所以当你区分一个具有实值的损失函数时，PyTorch给出了这个导数。


Pythorch如何计算 Wirtinger 共轭导数?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

通常，我们导数的公式以 `grad_output` 作为输入，表示我们已经得到的向量的Jacobian积，例如，
:math:`\frac{\partial L}{\partial s^*}`，这里 :math:`L` 是整个计算的损失函数（产生一个实损失值）
:math:`s`是这个函数的输出。
我们的目标是计算:math:`\frac{\partial L}{\partial z^*}`，这里 :math:`z`是函数的输入。
在实值损失函数的场景中，我们可以只计算:math:`\frac{\partial L}{\partial z^*}`。
总是链式法则暗示了我们也应该计算:math:`\frac{\partial L}{\partial z^*}`。
如果要跳过此推导，请查看本节中的最后一个公式，然后跳到下一节。



我们继续考虑在 :math:`f: ℂ → ℂ` 上的
:math:`f(z) = f(x+yj) = u(x, y) + v(x, y)j` 。
如前所述，Autograd的梯度约定是围绕实值损失函数的优化，因此我们假设 :math:`f` 是较大的实值损失函数 :math:`g` 的一部分。
使用链式法则，我们有：

    .. math::
        \frac{\partial L}{\partial z^*} = \frac{\partial L}{\partial u} * \frac{\partial u}{\partial z^*} + \frac{\partial L}{\partial v} * \frac{\partial v}{\partial z^*}
        :label: [1]

使用 Wirtinger 导数的定义，我们可以写出：

    .. math::
        \begin{aligned}
            \frac{\partial L}{\partial s} = 1/2 * (\frac{\partial L}{\partial u} - \frac{\partial L}{\partial v} j) \\
            \frac{\partial L}{\partial s^*} = 1/2 * (\frac{\partial L}{\partial u} + \frac{\partial L}{\partial v} j)
        \end{aligned}

注意到因为 :math:`u` 和 :math:`v` 是实值函数，
且根据我们的假设， :math:`L` 是实数， :math:`f` 是一个实值函数的一部分，我们有：

    .. math::
        (\frac{\partial L}{\partial s})^* = \frac{\partial L}{\partial s^*}
        :label: [2]

即， :math:`\frac{\partial L}{\partial s}` 等于 :math:`grad\_output^*`。

用:math:`\frac{\partial L}{\partial u}` 和 :math:`\frac{\partial L}{\partial v}` 解上面的不等式，我们有：

    .. math::
        \begin{aligned}
            \frac{\partial L}{\partial u} = \frac{\partial L}{\partial s} + \frac{\partial L}{\partial s^*} \\
            \frac{\partial L}{\partial v} = -1j * (\frac{\partial L}{\partial s} - \frac{\partial L}{\partial s^*})
        \end{aligned}
        :label: [3]

替换 :eq:`[1]` 中的 :eq:`[3]`，我们有：

    .. math::
        \begin{aligned}
            \frac{\partial L}{\partial z^*} &= (\frac{\partial L}{\partial s} + \frac{\partial L}{\partial s^*}) * \frac{\partial u}{\partial z^*} - 1j * (\frac{\partial L}{\partial s} - \frac{\partial L}{\partial s^*}) * \frac{\partial v}{\partial z^*}  \\
                                            &= \frac{\partial L}{\partial s} * (\frac{\partial u}{\partial z^*} + \frac{\partial v}{\partial z^*} j) + \frac{\partial L}{\partial s^*} * (\frac{\partial u}{\partial z^*} - \frac{\partial v}{\partial z^*} j)  \\
                                            &= \frac{\partial L}{\partial s^*} * \frac{\partial (u + vj)}{\partial z^*} + \frac{\partial L}{\partial s} * \frac{\partial (u + vj)^*}{\partial z^*}  \\
                                            &= \frac{\partial L}{\partial s} * \frac{\partial s}{\partial z^*} + \frac{\partial L}{\partial s^*} * \frac{\partial s^*}{\partial z^*}    \\
        \end{aligned}

使用 :eq:`[2]`，我们有：

    .. math::
        \begin{aligned}
            \frac{\partial L}{\partial z^*} &= (\frac{\partial L}{\partial s^*})^* * \frac{\partial s}{\partial z^*} + \frac{\partial L}{\partial s^*} * (\frac{\partial s}{\partial z})^*  \\
                                            &= \boxed{ (grad\_output)^* * \frac{\partial s}{\partial z^*} + grad\_output * {(\frac{\partial s}{\partial z})}^* }       \\
        \end{aligned}
        :label: [4]


最后一个公式是编写自己的梯度的重要公式，因为它将我们的导数公式分解为一个更简单的公式，易于手工计算。

如何为复函数写自己的求导公式
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

上面的方盒等式给出了复函数上计算导数的通用公式。
然而，我们仍需计算 :math:`\frac{\partial s}{\partial z}` 和 :math:`\frac{\partial s}{\partial z^*}`。
可以用以下两种方式计算：

    - 第一种方法是使用 Wirtinger 导数的定义来直接使用以下公式 :math:`\frac{\partial s}{\partial x}` 和 :math:`\frac{\partial s}{\partial y}`计算 :math:`\frac{\partial s}{\partial z}` 和 :math:`\frac{\partial s}{\partial z^*}`。
    - 第二种方法是用交换变量的小技巧，将 :math:`f(z)` 重写为两个变量函数 :math:`f(z, z^*)`，并通过把 :math:`z` 和 :math:`z^*` 作为不相关的变量来计算共轭导数。这通常很容易，因为如果函数是holomorphic的，只有 :math:`z` 才会被用到（ :math:`\frac{\partial s}{\partial z^*}` 的值会为零）。

考虑 :math:`f(z = x + yj) = c * z = c * (x+yj)` 函数，其中 :math:`c \in ℝ`.

使用第一种方法计算 Wirtinger 导数，我们有：

.. math::
    \begin{aligned}
        \frac{\partial s}{\partial z} &= 1/2 * (\frac{\partial s}{\partial x} - \frac{\partial s}{\partial y} j) \\
                                      &= 1/2 * (c - (c * 1j) * 1j)  \\
                                      &= c                          \\
        \\
        \\
        \frac{\partial s}{\partial z^*} &= 1/2 * (\frac{\partial s}{\partial x} + \frac{\partial s}{\partial y} j) \\
                                        &= 1/2 * (c + (c * 1j) * 1j)  \\
                                        &= 0                          \\
    \end{aligned}

使用 :eq:`[4]`, 和 `grad\_output = 1.0` (这是 :func:`backward` 在 PyTorch中被一个标量输出调用时的默认梯度输出)，我们有：

    .. math::
        \frac{\partial L}{\partial z^*} = 1 * 0 + 1 * c = c

使用第二种方法计算 Wirtinger 导数，我们直接得到：

    .. math::
        \begin{aligned}
           \frac{\partial s}{\partial z} &= \frac{\partial (c*z)}{\partial z}       \\
                                         &= c                                       \\
            \frac{\partial s}{\partial z^*} &= \frac{\partial (c*z)}{\partial z^*}       \\
                                         &= 0
        \end{aligned}

再次使用 :eq:`[4]` ，我们有 :math:`\frac{\partial L}{\partial z^*} = c`.
如你所见，第二种方法涉及的计算较少，并且对于更快的计算更为方便。

关于跨域函数(cross-domain functions)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

一些函数将复输入映射到实值输出。
这些函数是 :eq:`[4]` 的一个特例，我们可以通过下述链式规则推导出这个式子：


    - 对于 :math:`f: ℂ → ℝ`，我们有：

        .. math::
            \frac{\partial L}{\partial z^*} = 2 * grad\_output * \frac{\partial s}{\partial z^{*}}

    - 对于 :math:`f: ℝ → ℂ`，我们有：

        .. math::
            \frac{\partial L}{\partial z^*} = 2 * Re(grad\_out^* * \frac{\partial s}{\partial z^{*}})
