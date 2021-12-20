# 嵌入式AI -- 手写数字识别RISCV处理器C++纯底层实现(附带视频演示)

## 1 背景与前言

人类自产生以来，创造了辉煌的文明。例如、数学物理等惊人的理论成就，在这个过程中，阿拉伯数字起到了非常重要的作用。0 - 9 ，我们可以有各种各样的写法，不同的个体对每一个字符的写法都会根据他的书写特点而有所不同。相同个体的人，在自己不同的心情下写出来的数字形象很多时候也是截然不同。但是不变的是每一个数字字符，他的形状“大体很像”某个数字，让我们作为人类成功识别这些字符。


例如，下面的图片，同样是1，但是1的形象可能会稍有差别

![avatar](pic/1.png)

那么计算机能否做到这样“模糊”的识别呢？输入一张包含模糊数字形象的图片，计算机帮我们判断出这个数字是多少，并打印出来？

这个问题就是“手写数字识别”的问题，使用python在x86架构的计算机上使用tensorflow之类的框架可以非常轻松的几行搞定。但是我出于学习的目的，以及我最初想在资源非常有限的MCU上运行AI代码的[理想](https://zhuanlan.zhihu.com/p/439472022),于是写了此篇文章，记录使用C++在以RISC-V为核心的嵌入式处理器[BL706](https://www.bouffalolab.com/)上运行手写数字识别的例子。


## 2 基本算法与数据集

基本算法采用经典的双层BP神经网络，结构如下

![avatar](pic/2.png)

输入层维度选用28 * 28,对应是训练集每一张图片的像素大小，MINIST对应每一张图片的尺寸大小是28 * 28，因此输入层选用28 * 28 。

隐藏层神经元的个数选用200个，这个参数的大小是随意指定的，笔者试过，过多的隐藏层神经元并不会极大的提升准确率，并且会耗费非常更多的计算资源，这个参数会有一个最优选值。

输出层的维度选用十个神经元，因为需要预测图片为各个数字的概率，0-9对应十个数字，模型会给出这张图片对应为某个数字的概率。


**总体来说，需要做如下的事情**


![avatar](pic/4.png)

输入一个28 * 28 bytes的图片数据，通过模型计算得到它为某个数字的概率，MCU找到对应判断输出概率最大的数值。

那么，我们的训练集应该是这样的

![avatar](pic/5.png)

最好的情况是，当如输入训练集数据的时候，模型给出上图的结果。但是显然事实不应该是这样的，因为神经网络算法会模糊一些东西，增强联想的能力。所以最真实的情况是正确的数值概率非常接近于1，其他数值概率接近0。我们选取判断最大的数值为最终的输出结果。

**数据集合采用MINIST 免费数据集**

[MINIST](http://yann.lecun.com/exdb/mnist/)的数据包分为四个
分别为

train-labels-idx1-ubyte：训练集标签，每个标签为1个byte 包含60000个标签数据，前2个word包含文件信息，因此此文件大小为60000 + 8 Bytes。

train-images-idx3-ubyte：训练集图片数据，每个图片为28*28 bytes 包含60000个图片数据，前4个word包含文件信息，此文件大小为 60000 * 28 *28 + 16 Bytes。

t10k-labels-idx1-ubyte：训练集标签，每个标签为1个byte 包含10000个标签数据，前2个word包含文件信息，因此此文件大小为10000 + 8 Bytes。

t10k-images-idx3-ubyte：训练集图片数据，每个图片为28*28 bytes 包含10000个图片数据，前4个word包含文件信息，此文件大小为 10000 * 28 *28 + 16 Bytes。

了解其结构后，可以方便的使用C++预览数据集合。


## 3 C++实现

### 步骤1，导入训练集到特定的数据结构

``` C++
class label_data {
private:
    uint32_t fread_w(FILE *__restrict __stream)
    {
        uint32_t rlt = 0;
        uint8_t i = 0;
        int ch = 0;

        for (i = 0; i < 4; i++) {
            ch = fgetc(__stream);
            if (ch != EOF) {
                rlt <<= 8;
                rlt |= ch;
            }
        }
        return rlt;
    }

public:
    uint32_t magic_number;
    uint32_t items_number;

    uint8_t *label;

    label_data(const char *file_path)
    {
        FILE *fp = fopen(file_path, "r");
        uint32_t i = 0;

        magic_number = fread_w(fp);

        //printf("magic_number = 0x%x\r\n", magic_number);

        if (magic_number != 0x00000801) {
            printf("data set error\r\n");
            exit(0);
        }

        items_number = fread_w(fp);

        //printf("items_number = %d\r\n", items_number);

        label = new uint8_t[items_number];
        uint8_t tmp = 0;
        for (i = 0; i < items_number; i++) {
            tmp = fgetc(fp);
            label[i] = tmp;
        }
    }

    void display_label(uint16_t label_index)
    {
        uint8_t *label_fp = NULL;

        label_fp = label + label_index;

        printf("label[%d]=%d", label_index, *label_fp);
    }

    uint8_t *get_label(uint16_t label_index)
    {
        uint8_t *label_fp = NULL;

        label_fp = label + label_index;
        return label_fp;
    }
};

class image_data {
private:
    uint32_t fread_w(FILE *__restrict __stream)
    {
        uint32_t rlt = 0;
        uint8_t i = 0;
        int ch = 0;

        for (i = 0; i < 4; i++) {
            ch = fgetc(__stream);
            if (ch != EOF) {
                rlt <<= 8;
                rlt |= ch;
            }
        }
        return rlt;
    }

public:
    uint32_t magic_number;
    uint32_t image_number;
    uint32_t rows_number;
    uint32_t columns_number;

    uint8_t *image;

    image_data(const char *file_path)
    {
        FILE *fp = fopen(file_path, "r");
        uint32_t i = 0;

        magic_number = fread_w(fp);

        //printf("magic_number = 0x%x\r\n", magic_number);

        if (magic_number != 0x00000803) {
            printf("data set error\r\n");
            exit(0);
        }

        image_number = fread_w(fp);
        rows_number = fread_w(fp);
        columns_number = fread_w(fp);

        //printf("image size = %d\r\nrows_number = %d\r\ncolumns_number = %d\r\n", image_number, rows_number, columns_number);

        image = new uint8_t[image_number * rows_number * columns_number];
        uint8_t tmp = 0;
        for (i = 0; i < image_number * rows_number * columns_number; i++) {
            tmp = fgetc(fp);
            image[i] = tmp;
        }
    }

    void display_pic(uint16_t pic_index)
    {
        uint32_t i = 0, j = 0;
        uint8_t *image_fp = NULL;

        image_fp = image + pic_index * columns_number * rows_number;

        for (i = 0; i < rows_number; i++) {
            for (j = 0; j < columns_number; j++) {
                if (*(image_fp + i * columns_number + j) != 0) {
                    printf("0x%2x,", *(image_fp + i * columns_number + j));
                } else {
                    printf("     ");
                }
            }
            printf("\r\n");
        }
    }

    uint8_t *get_pic(uint16_t pic_index)
    {
        uint8_t *image_fp = NULL;

        image_fp = image + pic_index * columns_number * rows_number;

        return image_fp;
    }
};
```

上述代码创建了image与label的数据结构，用来妥善的把训练集数据与测试集数据用一个合理的数据结构包裹起来。

特地写了display函数，我们可以在控制台预览我们28*28的图片信息，查看它对应的标签数据。

``` C++
int main()
{
    image_data *train_image = new image_data("./train-images-idx3-ubyte");
    image_data *test_image = new image_data("./t10k-images-idx3-ubyte");

    label_data *train_label = new label_data("./train-labels-idx1-ubyte");
    label_data *test_label = new label_data("./t10k-labels-idx1-ubyte");

    train_image->display_pic(0);
    train_label->display_label(0);
}
```
输出的结果如下

```





                                                            0x 3,0x12,0x12,0x12,0x7e,0x88,0xaf,0x1a,0xa6,0xff,0xf7,0x7f,
                                        0x1e,0x24,0x5e,0x9a,0xaa,0xfd,0xfd,0xfd,0xfd,0xfd,0xe1,0xac,0xfd,0xf2,0xc3,0x40,
                                   0x31,0xee,0xfd,0xfd,0xfd,0xfd,0xfd,0xfd,0xfd,0xfd,0xfb,0x5d,0x52,0x52,0x38,0x27,
                                   0x12,0xdb,0xfd,0xfd,0xfd,0xfd,0xfd,0xc6,0xb6,0xf7,0xf1,
                                        0x50,0x9c,0x6b,0xfd,0xfd,0xcd,0x b,     0x2b,0x9a,
                                             0x e,0x 1,0x9a,0xfd,0x5a,
                                                       0x8b,0xfd,0xbe,0x 2,
                                                       0x b,0xbe,0xfd,0x46,
                                                            0x23,0xf1,0xe1,0xa0,0x6c,0x 1,
                                                                 0x51,0xf0,0xfd,0xfd,0x77,0x19,
                                                                      0x2d,0xba,0xfd,0xfd,0x96,0x1b,
                                                                           0x10,0x5d,0xfc,0xfd,0xbb,
                                                                                     0xf9,0xfd,0xf9,0x40,
                                                                      0x2e,0x82,0xb7,0xfd,0xfd,0xcf,0x 2,
                                                            0x27,0x94,0xe5,0xfd,0xfd,0xfd,0xfa,0xb6,
                                                  0x18,0x72,0xdd,0xfd,0xfd,0xfd,0xfd,0xc9,0x4e,
                                        0x17,0x42,0xd5,0xfd,0xfd,0xfd,0xfd,0xc6,0x51,0x 2,
                              0x12,0xab,0xdb,0xfd,0xfd,0xfd,0xfd,0xc3,0x50,0x 9,
                    0x37,0xac,0xe2,0xfd,0xfd,0xfd,0xfd,0xf4,0x85,0x b,
                    0x88,0xfd,0xfd,0xfd,0xd4,0x87,0x84,0x10,



label[0]=5
```

显然看到图片5的排布，以及其对应标签5，如此一来数据准备的工作就完成了。
### 步骤2，测试数学库工作是否正常

在PC平台，数学库使用的是[Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page)
在嵌入式平台，如果是ARM体系的处理器，数学库使用的是[CMSIS](https://www.arm.com/why-arm/technologies/cmsis)内标准的线性代数操作函数。
如果是RISCV体系的处理器架构，数学库使用的是[NMSIS](https://doc.nucleisys.com/nmsis/),本文的演示视频是使用的此数学库。

本文的神经网络算法仅仅用到了两个常用的数学库，两矩阵乘积，以及矩阵的转置。任意数学库都支持这两种操作，只是API名字不同而已。

测试代码如下

``` C++
void math_lib_test(void)
{
    MatrixXd a(2, 2);
    MatrixXd b(2, 2);
    MatrixXd c(2, 2);

    a(0, 0) = 1;
    b(0, 0) = 1;
    a(0, 1) = 2;
    b(0, 1) = 2;
    a(1, 0) = 3;
    b(1, 0) = 3;
    a(1, 1) = 4;
    b(1, 1) = 4;

    c = a * b;

    printf("[[%f,%f][%f,%f]] * [[%f,%f][%f,%f]] = [[%f,%f][%f,%f]\r\n]",
           a(0, 0), a(0, 1), a(1, 0), a(1, 1),
           b(0, 0), b(0, 1), b(1, 0), b(1, 1),
           c(0, 0), c(0, 1), c(1, 0), c(1, 1));

    c = a.transpose();

    printf("[[%f,%f][%f,%f]].transpose = [[%f,%f][%f,%f]\r\n]",
           a(0, 0), a(0, 1), a(1, 0), a(1, 1),
           c(0, 0), c(0, 1), c(1, 0), c(1, 1));
}
```

输出结果为

```
[[1.000000,2.000000][3.000000,4.000000]] * [[1.000000,2.000000][3.000000,4.000000]] = [[7.000000,10.000000][15.000000,22.000000]
][[1.000000,2.000000][3.000000,4.000000]].transpose = [[1.000000,3.000000][2.000000,4.000000]
```
### 步骤3，创建基于反向传播的训练算法，利用训练集进行训练

代码首先创建了一个trainer类型，抽象一个训练者。
创建训练者时，需要告知训练者神经网络的节点信息以及学习率。

下述代码构建了一个trainer的图纸，具体算法原理可以看我以前的《嵌入式AI xxx》系列文章。这里的激活函数选择的是sigmod函数，暂时还没有用到tanh或者relu，后面会逐步尝试。

权重初始化采用了C++的正态随即随方法，随机初始化了权重矩阵初始值。
``` C++
class trainer {
public:
    uint32_t inode;
    uint32_t hnode;
    uint32_t onode;
    double learnRate;

    //result
    MatrixXd wih;
    MatrixXd who;

    trainer(uint32_t _inode, uint32_t _hnode, uint32_t _onode, double _learnRate)
    {
        uint32_t i = 0, j = 0;
        inode = _inode;
        hnode = _hnode;
        onode = _onode;
        learnRate = _learnRate;

        /* wih and who are trainer's outputs*/
        /* create wih and who space */
        wih = MatrixXd::Zero(hnode, inode);
        who = MatrixXd::Zero(onode, hnode);

        /* random initial w */
        std::default_random_engine e(time(0));
        std::normal_distribution<double> n(0, 0.07);

        for (i = 0; i < (int)wih.rows(); i++) {
            for (j = 0; j < (int)wih.cols(); j++) {
                wih(i, j) = n(e);
            }
        }
        for (i = 0; i < (int)who.rows(); i++) {
            for (j = 0; j < (int)who.cols(); j++) {
                who(i, j) = n(e);
            }
        }
    }
    void shape(MatrixXd mat)
    {
        printf("rows = %d , cols = %d", (int)mat.rows(), (int)mat.cols());
    }
    void look(MatrixXd mat)
    {
        int i = 0, j = 0;
        for (i = 0; i < (int)mat.rows(); i++) {
            printf("\n");
            for (j = 0; j < (int)mat.cols(); j++) {
                printf("%lf ", mat(i, j));
            }
        }
    }

    double sigmod_function(double x)
    {
        return 1.0 / (1 + exp(-x));
    }

    uint8_t forecast(uint8_t *data)
    {
        /* forward-propagating calculate error */
        uint32_t i = 0, j = 0;
        uint8_t rlt = 0;
        double max = 0;
        /* set *data into martix data structure */
        MatrixXd dataM = MatrixXd::Zero(inode, 1);

        for (i = 0; i < inode; i++) {
            dataM(i, 0) = (double)((double)data[i] / (double)255.0 * (double)0.99) + (double)0.01;
        }

        MatrixXd a1 = MatrixXd::Zero(hnode, 1);
        MatrixXd a2 = MatrixXd::Zero(onode, 1);
        /* first layer propagating */
        a1 = wih * dataM;

        /* active */
        for (i = 0; i < a1.rows(); i++) {
            for (j = 0; j < a1.cols(); j++) {
                a1(i, j) = sigmod_function(a1(i, j));
            }
        }

        /* second layer propagating */
        a2 = who * a1;

        /* active */
        for (i = 0; i < a2.rows(); i++) {
            for (j = 0; j < a2.cols(); j++) {
                a2(i, j) = sigmod_function(a2(i, j));
                if (a2(i, j) > max) {
                    max = a2(i, j);
                    rlt = i;
                }
            }
        }
        return rlt;
    }

    /* study a data */
    double train(uint8_t *data, uint8_t *label)
    {
        double loss = 0;
        /* forward-propagating calculate error */
        uint32_t i = 0, j = 0;

        /* set *data into martix data structure */
        MatrixXd dataM = MatrixXd::Zero(inode, 1);

        for (i = 0; i < inode; i++) {
            dataM(i, 0) = (double)((double)data[i] / (double)255.0 * (double)0.99) + (double)0.01;
        }

        MatrixXd a1 = MatrixXd::Zero(hnode, 1);
        MatrixXd a2 = MatrixXd::Zero(onode, 1);
        /* first layer propagating */
        a1 = wih * dataM;

        /* active */
        for (i = 0; i < a1.rows(); i++) {
            for (j = 0; j < a1.cols(); j++) {
                a1(i, j) = sigmod_function(a1(i, j));
            }
        }

        /* second layer propagating */
        a2 = who * a1;

        /* active */
        for (i = 0; i < a2.rows(); i++) {
            for (j = 0; j < a2.cols(); j++) {
                a2(i, j) = sigmod_function(a2(i, j));
            }
        }

        /*BP*/

        /* set *label into martix data structure */
        MatrixXd labelM = MatrixXd::Zero(onode, 1);

        /* set label format into output format */
        for (i = 0; i < onode; i++) {
            labelM(i, 0) = 0.01;
        }
        labelM(*label, 0) = 0.99;

        /*  layer 2 error */
        MatrixXd delta2 = MatrixXd::Zero(onode, 1);

        for (i = 0; i < delta2.rows(); i++) {
            for (j = 0; j < delta2.cols(); j++) {
                delta2(i, j) = labelM(i, j) - a2(i, j);
                loss += delta2(i, j);
            }
        }

        /*  layer 1 error */
        MatrixXd delta1 = MatrixXd::Zero(hnode, 1);

        delta1 = who.transpose() * delta2;

        /* cal dwih */
        MatrixXd dwih = MatrixXd::Zero(hnode, inode);
        MatrixXd dwho = MatrixXd::Zero(onode, hnode);

        for (i = 0; i < delta2.rows(); i++) {
            for (j = 0; j < delta2.cols(); j++) {
                delta2(i, j) = delta2(i, j) * a2(i, j) * (1 - a2(i, j));
            }
        }

        dwho = delta2 * a1.transpose();

        //cal dwho
        for (i = 0; i < delta1.rows(); i++) {
            for (j = 0; j < delta1.cols(); j++) {
                delta1(i, j) = delta1(i, j) * a1(i, j) * (1 - a1(i, j));
            }
        }

        dwih = delta2 * dataM.transpose();

        //adjust wih and who
        for (i = 0; i < dwih.rows(); i++) {
            for (j = 0; j < dwih.cols(); j++) {
                wih(i, j) = wih(i, j) + dwih(i, j) * learnRate;
            }
        }
        for (i = 0; i < dwho.rows(); i++) {
            for (j = 0; j < dwho.cols(); j++) {
                who(i, j) = who(i, j) + dwho(i, j) * learnRate;
            }
        }
        printf("%f,%f ", wih(0, 0), who(0, 0));
        return loss;
    }
};
```

用图纸创建了一个具体的训练者实体，告知待训练的神经网络结构以及学习率。
``` C++

trainer *t = new trainer(28 * 28, 200, 10, 0.05);

```

然后把我们之前封装好的训练集数据一个一个的交给训练者，让他一张一张的学习。
这里的epochs代表着训练者需要把整个训练集学习几次。训练集就像一本画册，训练者一张一张读完一遍之后，害怕自己忘记，再多读一遍。巩固记忆，这里的epochs就是复习的次数。实验看下来不是复习次数越多越好的。这里选择复习两次就好。
``` C++
for (tim = 0; tim < epochs; tim++) {
    for (i = 0; i < train_image->image_number; i++) {
        printf("loss=%lf %d/%d(%d)\r\n", t->train(train_image->get_pic(i), train_label->get_label(i)), i, train_image->image_number, tim);
    }
}
```

### 步骤4，用测试集查看模型准确性


学习完成之后，通过测试集给训练者做一次期末考试，给他一个它之前从未见过的新图片，它猜出这个值是多少，然后对比下是否和正确答案一致，如果一致则加分，否则不加分。

``` C++
uint32_t right_num = 0;
for (i = 0; i < test_image->image_number; i++) {
    label = *test_label->get_label(i);
    forcast = t->forecast(test_image->get_pic(i));
    if (label == forcast) {
        right_num++;
    } else {
        printf("correct = %d , forecast = %d error\r\n", label, forcast);
    }
}
```

**最终输出的结果是**

```
correct = 8699 , error = 1301 rate = 0.869900
```
总共10000个测试数据，模型正确了86%左右。

## 4 结果分析

再python环境下，使用numpy库以及完全相同的算法，我得到的准确率是在96%左右，非常高，但是为什么C++环境下准确率下降了呢？

我一直在查找算法中存在的问题，但是最终还是没有找到实现上的问题，而且其他和我有一样经历的博主用C++实现出来也是88%左右的识别率。因此也增强了我对实现上没问题的信心。

后来想到，很大可能是C++浮点数据结构的精度决定的。Python环境默认的数据精度是float64,C++使用的是double数据类型。

通过查阅资料发现python的浮点数最大精度是支持到17位[最大精度是支持到17位](https://www.cnblogs.com/herbert/p/3402245.html)而C++的double类型[最大精度仅支持到16位](https://blog.csdn.net/black_kyatu/article/details/79257346)。

因此，理解到很大可能是最底层的计算精度影响了python环境与C++环境在相同算法上的识别率。

这个问题存档，后期再做卷积神经网络相关的实验的时候进一步验证此猜想。

**如果您有问题的答案也请再评论区告诉笔者。**

# 5 嵌入式平台移植

RISC-V是一个开源社区维护的CPU处理器架构，具体的信息可以看我之前的文章[用一段C代码编译的指令代码，来阐明RISC-V架构的简洁](https://zhuanlan.zhihu.com/p/237357630)等等。

BL706是基于RISC-V处理器架构，由博流智能推出的的一款蓝牙、zigebee无线射频芯片。当然它也包含了各种常用的嵌入式外设（SPI I2C...等等）。

感兴趣的朋友可以到博流智能维护的[开源社区](https://gitee.com/bouffalolab/bl_mcu_sdk)了解更多。

笔者在这个资源有限的平台上使用C/C++从最底层实现了字迹识别的功能。

## 硬件准备

一块博流智能推出的bl706_AVB开发板

![avatar](pic/6.png)

一个常用的TFT屏幕

![avatar](pic/7.png)

## 总体流程与视频演示

由于嵌入式平台的ROM空间有限，因此训练环节不能在嵌入式平台进行，所以训练环境在PC环节使用C++进行，将训练出的参数导出为.h头文件格式，给到嵌入式平台使用。

使用SPI硬件控制器直接驱动TFT显示屏，读取液晶显示屏的输入结果，实现一个手写的功能，嵌入式设备将用户的手写轨迹记录在一个[28 * 8 , 28 * 8]的矩阵中。

之后对手写轨迹的图片矩阵进行压缩，从[28 * 8 , 28 * 8]压缩为[28,28]。因为训练模型的图片大小是[28,28]，这个对于人来说太小了，写起来很小，因此总体目的是让用户写一个较大图片，然后后期压缩为模型可以输入的大小图片。

压缩完成图片信息之后直接输入到已经训练好的神经网络，嵌入式设备完成识别，并打印在屏幕上。

**视频如下**

https://www.zhihu.com/zvideo/1456306513770319873

**可以看到识别率还是非常高的**

TinyAi的开源项目
https://github.com/JingyanChen/TinyAi