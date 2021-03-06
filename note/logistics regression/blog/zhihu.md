# 嵌入式AI -- Logistics Regression二分类问题

## 前言

无论是身处在温暖的家中，惬意的呼喊“天猫精灵”、“小爱同学”，还是在街边摊点拿着兼并油条准备付钱时的刷脸支付，机器学习已经俨然成为了我们生活中不可或缺的一部分。绝大多数机器学习的方法，都是以框架+高级脚本语言的形式使用的，这些框架和高级脚本语言需要非常昂贵的运行环境，一台装有linux或者windows操作系统的台式机，或者类似树莓派这样的小型计算机，很多时候需要联网。流行的机器学习算法在装载着linux的x86处理器架构运行似乎不是一件难事，选择tensorflow或者potych框架，然后根据框架服务商提供的Tutorial就可以，很快就能部署起非常酷炫的AI应用，例如人形识别之类。

![avatar](app.png)

然而在众多应用场景，例如在网络条件很差的环境，设备不能将传感器数据发送到远端AI服务器，或者场景不允许昂贵的设备条件。在这些应用场景下，固有的流行框架+高级脚本语言方案就不能满足要求。

这些应用场合的最佳的计算机设备就是MCU，我会把它称之为嵌入式设备，他的特点是小巧，低功耗，低成本，但是相应的ram和rom资源较少，网络环境配置难度也比普通平台大。

笔者了解到大厂在将他们固有的框架慢慢移植进MCU领域，例如[tensorflow Lite](https://tensorflow.google.cn/lite/microcontrollers/overview)，但是笔者发现这个代码依旧非常臃肿，看不出嵌入式应有的简洁，框架就是简化版的tensorflow，有很多遗留问题。

笔者有一个理想，希望首先将一些经典的机器学习方法，使用C/C++来实现一遍，例如神经网络。笔者通过网络环境学习到了一些经典的机器学习理论，希望这些算法能以轻量的方式运行在各种不同平台的MCU中，使用C/C++语言的优点是可移植性很强，可以在任意平台编译运行，不需要部署任何框架环境，只需要安装好对应处理器架构的gcc Toolchain。

我在github托管了一个名为TinyAi的开源项目，记录我这个理想的实现过程，希望和大家能有所交流。

https://github.com/JingyanChen/TinyAi

这个项目的初衷是自己学习使用，最终的目标是将这些算法运行在各大MCU平台上。

这个系列《嵌入式AI--xxx》文章是开源项目的其中一部分，所有源码都可以在仓库中找到。

# Logistics Regression
## 从提出问题开始

![avatar](../pic/pic1.png)

假设在x1,x2两个维度确定的平面里，有m个小球，红色小球(叉叉形状)以及蓝色小球（圈圈形状）。是否可以画出一条线，将红球和黑球区分开来。

换一种提法，当出现了一个新的球(颜色未知)，随机扔出去(落在平面内会确定一个坐标(x1,x2)),那么请问他是红球的概率是多少？

这些提法用数学的表达就是

假设已有数据集  
$$D =\{ (x^{(0)},y^{(0)}),(x^{(1)},y^{(1)}),..(x^{(m)},y^{(m)}) \} \ x \in R^n \: y^{(i)} \in \{0,1 \}$$

$x^{(i)}$是n维度样本数据，在此问题中是二维的维度也就是$x^{(i)}={({x}^{(i)}_1,{x}^{(i)}_2)}$，$y^{(i)}$是标签,在此问题中是1或者0，如果样本的$y^{(i)}$是1可以理解为是红球/点落在边界上方，如果样本的$y^{(i)}$是0可以理解为是蓝球/点落在边界下方。

**我们的目的是确认一个预测函数**

$$\hat{y}=f(x ,w)$$

使得当我出现了一个新的小球落在(x1,x2)上,预测其是红色/落在边界上方的概率
$$\hat{y}=P(y=1|x) = f(x ,w)$$
x是新的球的坐标，w是模型参数，$\hat{y}$是新球是红球的概率，f是预测函数，暂时还没有确定。

### 建立模型的过程

根据上述的目标，我们有两个任务要做

1.确定预测函数f的基本结构
2.确定预测函数f里的参数具体的最优值

首先，先做第一个任务，确认预测函数f的基本结构。

我们用一个线性变换，将概率$\hat{y}$与二维数据x=(x1,x2)联系起来。

$$z = w1 * x1 + w2 * x2 + b$$

这里用三个参数(w1,w2,b)来联系了预测概率z与二维数据x=(x1,x2)之间的关系。

但是出现一个显然的问题，概率是一个[0,1]数值，如果不在这个区间，那么就很难使用概率的思想。

因此需要对z进行一个sigmoid变换。

$$\sigma(z) =\frac{1}{1+e^{-w^{T}z}}$$

![avatar](sigama.png)

使用sigmoid函数之后，可以把任意值域的函数转化为[0,1]范围，他的优点很多，暂时先不提，至少帮助我们把$$z = w1 * x1 + w2 * x2 + b$$中的 $z$ 通过 
$$a = \sigma(z)$$
转化为 $a$,$a\in(0,1)$。

那么，我们就选择这个函数作为我们的预测函数

$$\hat{y}=P(y=1|x) = \sigma(w1 * x1 + w2 * x2 + b)$$

至此，我们的任务一完成了，找到了一个预测函数的基本结构。

接下来，我们思考的重点变为了如何确定这个预测函数的参数 w1,w2,b，哪一组参数是最佳的模型参数？

### 极大似然估计的思想

一般来说，当我们已知概率模型的参数 $\Theta$ 然后我们把事件x输入概率函数，就可以计算出事件x发生的概率,这个时概率计算问题。$$P(x|\Theta)$$

但是，当我们通过已知的样本，反推其模型参数$\Theta$时，这个就是极大似然估计的问题。
$$P(\Theta|x)$$

我个人的总结如下，极大似然估计的思想关键在于，你固定了一组模型参数$\Theta$从而确定了一个概率函数来预测未发生的事件概率，那么如何评判你的这个$\Theta$参数配置的好不好？我们不能拿未知的事情来评判，只能从已经发生的事件来验证，而验证的方法简而言之就是“预测出来的所有事件同时发生的概率最大”，越大说明模型参数越好。

从本章最初的例子来说，假定我已经拥有了一些“事实”数据集合 
$$D =\{(0,0,1),(1,2,0),(3,5,1)\}$$ 

其中括号里的数据意义时(x1,x2,y),(x1,x2)组成了小球所在的坐标,y为1代表为红球，0代表为黑球。

那么，假设我建立了一个概率预测函数,预测出来一个新的坐标为(xx1,xx2)的小球，他为红球的概率。
$$P(y=1|x)$$

那么，我们先不谈预测未知的事件，我们肯定希望已经发生的事情经过你的预测函数与事实结果一致！

我们希望事件 {坐标为(0,0)的球是红球} {坐标为(1,2)的球是蓝球} {坐标为(3,5)的球是红球} 这三个事件带入我们的预测函数，同时发生的概率最大。因为这三个事件同时发生是我们样本的事实，所以这个是我们评判参数是否合理的重要依据。

也就是

$$L=P({坐标为(0,0)的球是红球})P({坐标为(1,2)的球是蓝球})P({坐标为(3,5)的球是红球})$$
L越大，说明预测函数 $P(x|\Theta)$越好。

### 回到问题本身

我们之前确定了一个预测函数，这个函数基本的结构我们确定好了，但是部分参数还需要确认。

$$p(x) = P(y=1|x) = \sigma(w1 * x1 + w2 * x2 + b) $$

我们根据极大似然估计的思想，我们拥有m个样本点，他们包含了球的坐标$x^{(i)}=({x^{(i)}}_0,{x^{(i)}}_1)$,同时也包含了这些样本点事实的颜色输出结果 $y^{(i)}=\{0 \ or \ 1\}$

那么我们建立一个极大似然估计函数

$$L = \Pi(p(x)^{y^{(i)}} * (1-p(x))^{(1-y^{(i)})}) $$

L是所有样本同时发生的概率，是每个独立事件的连乘。

假设第一个样本输出结果是1时，$$L=p(x) * ... $$

当第二个样本输出结果是 0 时,

$$L=p(x) * (1-p(x)) * ...$$

以此类推。

读者可以自行思考下L函数是否是所有样本事件同时发生的联合概率。

为后续求导方便，我们会对式子两边同时做ln操作，将连乘转化为连加

$$L = \Sigma(y^{(i)}Inp(x) + (1-y^{(i)})In(1-p(x))) $$

$$ = \Sigma(y^{(i)}In\frac{p(x)}{1-p(x)} + In(1-p(x))) $$

那么根据上一节的理论，我们希望L越大越好，越大说明参数选择越好。但是在机器学习中我们更喜欢谈论越小越好的问题，所以干脆定义一个损失函数J。

$$J=-L$$

这样，我们定义了损失函数J，就可以把问题变化为，如何选择合适的(w1,w2,b)模型参数，使得损失函数J最小。

## 如何获得最佳参数使得损失函数最小

目前我们完成了任务1，确认好了预测函数的基本结构$\hat{y}=\sigma(x1*w1+x2*w2+b)$
并且我们知道了评判模型参数(w1,w2,b)好坏的依据是损失函数J尽可能的小。

那么确定最佳参数的问题变为了求极值的问题。我们一般会使用梯度下降法来求解此问题。

针对目前待解决的问题，梯度下降法的结论是，

当(w1,w2,b)的变化方向如下所示的时候 

$$w1 = w1 - \frac{\partial J}{\partial w1} $$

$$w2 = w2 - \frac{\partial J}{\partial w2} $$

$$b = b - \frac{\partial J}{\partial b} $$

损失函数J会越来越小。理论证明会在后续文章出给出，这篇文章大可先确认这个结论。

于是我们找到了确定(w1,w2,b)参数的方法
$$0:给定初始值w1=0,w2=0,b=0$$ $$\downarrow$$ $$1:z=w1 * x1 + w2 * x2 + b$$$$\downarrow$$$$2:\hat{y}=a=\sigma(z)$$$$\downarrow$$$$3:计算损失函数 L = \Sigma(y^{(i)}Inp(x) + (1-y^{(i)})In(1-p(x))) $$$$\downarrow$$$$4:根据梯度下降法更新w1,w2,b三个参数$$$$\downarrow$$$$回到步骤1，循环迭代直到迭代结束$$

## C/C++实现

###  准备数据集合

笔者在网上找到了一个样本数为100的训练数据，存放在远程仓库../data/文件夹下，文件预览如下
```
-1.510047	6.061992	0
-1.076637	-3.181888	1
1.821096	10.283990	0
3.010150	8.401766	1
-1.099458	1.688274	1
-0.834872	-1.733869	1
-0.846637	3.849075	1
1.400102	12.628781	0
1.752842	5.468166	1
0.078557	0.059736	1
0.089392	-0.715300	1
```

为了C++处理文件信息方便，要求数据的排列时 x1(0x09)x2(0x09)y(0x0d0x0a)其中x1 x2时输入的坐标值 y是对应的样本输出，0x09是横向制表符，0x0d0x0a是换行符。

笔者为这样的文件结构，提供了一套解析代码，可以快速将这些数据填充进入矩阵数据结构。

模型评优的方法，这里选择了“留出法”。在原有100个样本中，选择80个作为训练集，留下20个作为测试集合。通过20个测试集合判定模型好坏。

![avatar](data.png)

如果让你画一条**直线**来区分红球和绿球，你可以做到吗？
### 矩阵操作库函数

考虑到机器学习应用会使用到大量的矩阵运算，需要选择一个开源的矩阵运算库，在选择的过程中考虑到下述两点原因
1. 可移植性，笔者不希望看到大量的.c/.h依赖，最好能做到拿来就用
2. 代码大小，嵌入式平台的rom与ram资源有限
3. 需要为支持SIMD预先做准备

综合上述原因，我选择了[Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page),他可以用来进行线性代数、矩阵、向量操作等运算，他最大的亮点在于只需要包含对应头文件就可以调用其矩阵运算函数。

已经将其移植在component\eigen目录下。

### 核心代码
``` C++
class logisticsRegression {
    public:
    MatrixXd data;  /*(m * n) data matrix */
    MatrixXd label; /*(m * 1) label matrix */
    MatrixXd w;     /* (w0,w1,...) model parameter*/

    int n; /* dimensionality */
    int m; /* sample number*/

    logisticsRegression(const char *__restrict file_url){}
    void train(double lamabda, int k){}
}
```
在构造函数里，通过输入文件URL来创建对应大小的矩阵空间，以及导入数据到数据data以及对应标签label。

当数据填充完毕后，使用train函数开始训练，输入参数学习率lamabda，以及总的迭代次数k。

这里测试的经验看到当学习率过大的时候，会出现损失函数来回震荡，一会正数，一会负数。类似于我们在两个山峰之间，中间是山谷，我们想到山谷下面，步子迈的太大了，一脚迈到了对面的山谷，发现不对劲，又一脚迈回原点山谷。此时我们只要把步子迈小一点，就可以一步步往山谷走。

### 训练过程节选
学习率设置为0.001 总迭代次数设定为100000,初始w0= 0，w1=1，b=0。使用的是批量梯度下降法，把所有样本的梯度求和作为下一步参数更新的依据。
```
    w0= 0.000000 , w1= 0.000000 , b= 0.000000 , loss sum = 55.451774
    w0= 0.001231 , w1= -0.138220 , b= 0.001000 , loss sum = 48.088690
    w0= 0.004241 , w1= -0.118777 , b= 0.018616 , loss sum = 47.590475
    w0= 0.006972 , w1= -0.118246 , b= 0.034081 , loss sum = 47.346577
    w0= 0.009648 , w1= -0.119827 , b= 0.049239 , loss sum = 47.107765
    w0= 0.012295 , w1= -0.121498 , b= 0.064311 , loss sum = 46.871408

    ...

    w0= 0.823328 , w1= -2.145026 , b= 16.114067 , loss sum = 7.427362
    w0= 0.823329 , w1= -2.145027 , b= 16.114074 , loss sum = 7.427361
    w0= 0.823329 , w1= -2.145028 , b= 16.114081 , loss sum = 7.427361
    w0= 0.823329 , w1= -2.145029 , b= 16.114087 , loss sum = 7.427361
    w0= 0.823329 , w1= -2.145030 , b= 16.114094 , loss sum = 7.427361
    w0= 0.823329 , w1= -2.145031 , b= 16.114101 , loss sum = 7.427361

```
可以看到，程序从我们最初预定的初始w值开始，一步步去尝试新的w，目标是损失最小。
从结果可以看出梯度下降法是正确的，他提供的w变化方向的确是损失函数减小的方向。
100000次后损失减小的越来越不明显，这边认为基本上是训练完了，得到了我需要三个最佳模型参数 w0 w1 b。

### 观察下模型训练的怎么样

之前我们留下了20个测试数据，我们将待测试的x1,x2带入我们的预测函数
$$\hat{y}=P(y=1|x) = \sigma(w1 * x1 + w2 * x2 + b)$$
我们上述做了非常多的工作最终确定了最佳的w1 w2 b参数
$$\hat{y}=P(y=1|x) = \sigma(0.823329 * x1 + -2.145031 * x2 + 16.114101)$$

计算出的结果如下
```
    forecast = 0.866190 real = 0.000000 error = 0.866190
    forecast = 1.000000 real = 1.000000 error = -0.000000
    forecast = 0.011589 real = 0.000000 error = 0.011589
    forecast = 0.638856 real = 1.000000 error = -0.361144
    forecast = 0.999991 real = 1.000000 error = -0.000009
    forecast = 1.000000 real = 1.000000 error = -0.000000
    forecast = 0.999224 real = 1.000000 error = -0.000776
    forecast = 0.000054 real = 0.000000 error = 0.000054
    forecast = 0.997064 real = 1.000000 error = -0.002936
    forecast = 1.000000 real = 1.000000 error = -0.000000
    forecast = 1.000000 real = 1.000000 error = -0.000000
    forecast = 0.000067 real = 0.000000 error = 0.000067
    forecast = 0.009700 real = 0.000000 error = 0.009700
    forecast = 0.999999 real = 1.000000 error = -0.000001
    forecast = 0.999998 real = 1.000000 error = -0.000002
    forecast = 0.999986 real = 1.000000 error = -0.000014
    forecast = 0.002030 real = 0.000000 error = 0.002030
    forecast = 0.999999 real = 1.000000 error = -0.000001
    forecast = 0.058333 real = 0.000000 error = 0.058333
    forecast = 0.000000 real = 0.000000 error = 0.000000
```

看出来第一个数据就拉跨了，事实上的样本输出是0，但是居然预测除了0.866的值，相差很大。这里应该不能太强求机器做出100%的决策，而且用直线来区分两类球的确非常困难，人也很难画出来？不信可以到“准备数据集合”那边看着图像想象一下？

除了第一个数据之外，其他数据都是非常好的预测了样本 事实为1的基本上预测出来的数据也是接近1，事实为0的预测结果也是接近0，从直观感觉上来看训练时非常成功的！

**回到起点，我们解决了最初提出的问题吗？**


# 小结

通过 吴恩达教授的[《深度学习》](https://www.bilibili.com/video/BV1FT4y1E74V?p=32)课程，我学习到了逻辑回归其实就是一个简单的神经元，以上过程使用C/C++实现了逻辑回归实质上就是完成了单个神经元的训练与测试，这给我带来极大成就感，为我将经典AI算法移植到MCU的大目标离得更进一步。当然后面还有许多困难，但是我相信一定可以完成的。如果大家喜欢这篇文章，或者希望和我一起来做这件事，请多多关注我，关注下[TinyAI](https://github.com/JingyanChen/TinyAi)仓库的最新进展。

感谢你阅读到这，愿君平安！