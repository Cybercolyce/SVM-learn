## 零·引言

**Support Vector Machine,SVM**是一种用于分类的算法，从最简单的二分类问题到多分类问题，他都有很好的实现效果。它的实现理念是这样子的：

> 支持向量机（Support Vector Machine, SVM）是一类按<font color="red">监督学习</font>方式对数据进行二元分类的广义线性分类器，其决策边界是对学习样本求解的最大边距<font color="red">超平面</font>。
>
> <p align="right">
>     ——百度百科
> </p>

先进行一些名词的解释：

#### 监督学习

> 所谓监督学习，是指给定一组<font color='red'>已知类别的样本</font>，同时给定指标以及分类标签，调整分类器的参数，让它达到所要求的性能。与之相对的非监督学习则是不给定分类标签，仅让机器自行分析、推断数据的内在结构。简单来说，<font color='red'>是否有人为给定评判标准</font>是区别监督与非监督的一个区别。

#### 超平面

> 所谓超平面，就是 n 维线性空间中维度为 n-1 的子空间，它可以把线性空间分割成不相交的两部分。比如二维空间中，一条直线是一维的，它把平面分成了两块；三维空间中，一个平面是二维的，它把空间分成了两块。



所以SVM的使用可以简单概括为：将一组给定标签的数据，寻找超平面将数据分类，使得不同类别的数据之间间隔最大。SVM的核心思想，就是将不同类的数据间隔扩大。



## 一 · 简单的SVM模型搭建

## 1.1 线性的二维SVM模型

### 1.1.1 线性可分

如图所示，左边是线性不可分，右边是线性可分的。简单来说，在二维空间中，能用一条线把两类点分离开那么就可以称为线性可分。

![image-20210509113920011](https://i.loli.net/2021/05/09/UIManQyDt82q3CK.png)

<center><font color='grey'>图1，非线性与线性可分</font></center>



### 1.1.2 支持向量

给定线性可分的训练样本集$T=\{(x_1,y_1),(x_2,y_2),...,(x_m,y_m)\},y_i\in{\{1,-1\}}$后，线性分类器将基于训练样本$T$在二维空间中找到一个超平面来正确划分两类样本。显然，满足条件的超平面有很多，而我们旨在找到<font color='red'>最好的</font>。

> 所谓最好的，是指我们要找到与两类数据间隔最大的超平面，使之对训练集之外的数据或噪声有最大的“容忍”能力。因为在训练过程中会因为训练集的局限性以及噪声因素会影响超平面的划分，所以我们要找到最能“容忍”噪声的那个超平面。

先定义距离的概念。我们使用欧氏几何距离：
$$
给定直线l:\\
l:Ax+By+C=0\\
则\\
二维平面的点到线距离d:\\
d=\frac{|Ax_p+By_p+C|}{\sqrt{A^2+B^2}}
$$
在这里我们要清楚一个事实：**点是由线构成的。**因此要用一条线把两类点分离的话，我们应该<font color='red'>从两类点距离最近</font>的地方开始下手，因为这个地方的点集成为最终划分直线上的点的概率最大。我们把这些起重要的参考作用的点称为<font color='navyblue'>**“支持向量”**</font>**（Supported Vector）**

<img src="https://upload-images.jianshu.io/upload_images/22276164-df81201d2150a248.jpg" alt="img" style="zoom:50%;" />

<center><font color='grey'>图2，支持向量</font></center>

## 1.2 SVM的数学公式推导

在图2中我们可以看到有四个标红的点，他们都是支持向量。他们确定了两条虚线，我们称为“**支撑超平面**”，并将他们的**间隔用$\gamma$表示**。不难看出当间隔越大，容忍噪声的能力越好，因为我们最终确定的直线是在间隔的**''中间''**位置。因此我们希望间隔越大越好！
$$
J(A,B,C,i)=arg_{A,B,C}maxmin(d_i)
$$
$d_i$表示样本点$i$到某条固定分割面的距离；$min(d_i)$表示所有样本点中与某个分割面之间距离的最小值，$arg_{A,B,C}maxmin(d_i)$表示从所有分割面寻找$\gamma$最大的超平面，其中$A，B，C$是分割面的参数。接下来我们进行模型的量化操作。

### 1.2.1$\gamma$的表示

不妨从最简单的二分类问题入手，见图2。定义实心点为正例，空心点为负例，分别用+1与-1表示。现在我们假设存在这么一条由SVM确定的直线$l:Ax+By+C=0$，显然所有的点点带入直线方程后它的绝对值一定大于1，因此我们不妨将$\gamma$定义如下：
$$
\hat\gamma_l=y_i'\times(Ax_i+By_i+C)
$$
其中$y_i'$代表样本点所属的类别，用+1和-1表示。当乘号右边的式子计算结果大于等于1时，分割面将样本点归类为正例，且**值越大于1则为正例的概率越高**；当乘号右边的式子计算结果为小于等于-1时，归类为负例，且**值越小于-1为负例的概率越高**。但是这种写法会带来一个问题，就是当A、B与C同比例改变时，直线是没有发生改变的，但乘号右边的值却被改变了：
$$
Ax+By+C=0\\

2Ax+2By+2C=0\\
$$
上二式表示的为同一条直线，但当带入点到$\hat\gamma$的表达式后，乘号的右边被放大了两倍，会造成误判。因此我们需要对$\hat\gamma$进行归一化处理：
$$
\gamma=\frac{\hat\gamma}{\sqrt{A^2+B^2}}
$$

### 1.2.2 目标函数的改写

可以发现这个$\gamma$表达式正好和二维平面中点到线距离是相同的，因此我们的目标函数可以改写成：
$$
J(A,B,C,d_i)=arg_{A,B,C}max\frac{1}{\sqrt{A^2+B^2}}min(\hat\gamma)
$$
而我们知道$\hat\gamma\geq1$,因此$min(\hat\gamma)=1$,从而目标函数可以改写成如下形式：
$$
\begin {cases}
\max{\frac{1}{\sqrt{A^2+B^2}}}\\
s.t. y_i'\times(Ax_i+By_i+C)\geq1
\end {cases}
$$
求$\max{\frac{1}{\sqrt{A^2+B^2}}}$其实是与求$\min{\frac{1}{2}({A^2+B^2})}$等价的，因此再次重新表示为：
$$
\begin {cases}
\min{\frac{1}{2}({A^2+B^2})}\\
s.t. y_i'\times(Ax_i+By_i+C)\geq1
\end {cases}
$$
现在我们将目标函数改写成一个给定不等式约束求$\frac{1}{2}({A^2+B^2})$的最小值问题。

### 1.2.3 拉格朗日乘子法求解问题（KKT对偶求解）

拉格朗日乘子法是用于求解给定$g(x)\leq0$约束时求一个$f(x)$最小值的方法。如需得到最优的解则需要用拉格朗日对偶性将问题转换为对偶问题，即：
$$
min(f(x))=minx_xmax_\lambda (L(x,\lambda))\\
=minx_xmax_\lambda(f(x)+\sum_{i=1}^{k}\lambda_i g_i(x))
$$
其中，$f(x)+\sum_{i=1}^{k}\lambda_i g_i(x)$为拉格朗日函数，$\lambda_i$为拉格朗日乘子，它是一个大于0的数。再通过对偶性，将问题转换为极大极小问题：
$$
min(f(x))=max_\lambda minx_x (L(x,\lambda))\\
=max_\lambda min_x(f(x)+\sum_{i=1}^{k}\lambda_i g_i(x))
$$

所谓对偶性，指的是一个优化问题可以从两个角度来看：原问题与对偶问题。一个原问题可以有很多个对偶问题，而只要找出一个对偶问题的解就能间接得到原问题的解。

最后我们的目标函数最终表达式改写为：
$$
\frac{1}{2}({A^2+B^2})=max_\lambda min_{A,B,C}(L(A,B,C,\lambda_i))\\
=max_\lambda min_{A,B,C}[\frac{1}{2}(A^2+B^2)-\sum_{i=1}^{n}a_iy_i'\times(Ax+By+C)+\sum_{i=1}^{n}\lambda_i]
$$
先求解$\min_{A,B,C}L(A,B,C,\lambda_i)$。对函数$L(A,B,C,\lambda_i)$求偏导并令导函数为0，最后将结果带入到目标函数的最终表达式中可得到：
$$
\min{\frac{1}{2}(A^2+B^2)}=max_\lambda(\frac{1}{2}(A^2+B^2)+\sum_{i=1}^{n}\lambda_i(1-y_i'\times(Ax_i+By_i+C)))\\
=max_\lambda [-\frac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{n}\lambda_i\lambda_jy_i'y_j'(x_i·x_j)+\sum_{i=1}^{n}\lambda_i-0]
$$
即：
$$
\begin{cases}
min_\lambda(\frac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{n}\lambda_i\lambda_j y_i'y_j'(x_i·x_j)-\sum_{i=1}^{n}\lambda_i)\\
s.t.\sum_{i=1}^{n}\lambda_i y_i'=0
\end{cases}
$$


因此我们得到我们的分割面参数的求解结果：
$$
\begin {cases}
(A,B)=\sum_{i=1}^{n}\lambda_ly_i'x_i\\
C=y_j-\sum)_{i=1}^{n}\lambda_ly_i'(x_i·y_j)
\end {cases}
$$
z注意求解C的时候需要固定某个$y_j$，需要从诸多拉格朗日乘子中选择一个大于0的j样本与后面的求和式相减。



## 二 · 一个简单的例子应用

给定如下的二维平面点，利用SVM进行分类,确定划分的超平面（二维平面中即直线）：

![image-20210509191631050](https://i.loli.net/2021/05/09/oJCjr3VnAYd69cG.png)

### 第一步

计算目标函数的最小值：
$$
min\frac{1}{2}(A^2+B^2)=f(\lambda)\\
\\=min_\lambda[\frac{1}{2}(18\lambda_1^2+25\lambda_2^2+2\lambda_3^2+42\lambda_1\lambda_2-12\lambda_1\lambda_3-14\lambda_2\lambda_3)-\lambda_1-\lambda_2-\lambda_3]
$$
同时$\lambda_i$满足如下关系：
$$
\lambda_1+\lambda_2-\lambda_3=0
$$

### 第二步

将该式子带入进最小值方程中有:
$$
min\frac{1}{2}(A^2+B^2)=f(\lambda)=\\
min_\lambda[\frac{1}{2}(18\lambda_1^2+25\lambda_2^2+2\lambda_3^2+42\lambda_1\lambda_2-12\lambda_1\lambda_3-14\lambda_2\lambda_3)-\lambda_1-\lambda_2-\lambda_3]=\\
min_\lambda(4\lambda_1^2+\frac{13}{2}\lambda_2^2+10\lambda_1\lambda_2 -2\lambda_1-2\lambda_2)
$$

### 第三步

对$\lambda_i$求偏导，并令导函数为0
$$
\begin{cases}
\frac{\partial f}{\partial \lambda_1}=8\lambda_1+10\lambda_2-2=0\\
\frac{\partial f}{\partial \lambda_2}=13\lambda_2+10\lambda_1-2=0
\end{cases}
$$
解方程可得$\lambda_1=\frac{3}{2}$,$\lambda_2=-1$。显然$\lambda_2$不满足$\lambda_i\geq0$的条件，因此我们需要令$\lambda_2=0$或$\lambda_1=0$再带入回最小值方程再次求解。

##### 情况一：

当$\lambda_1=0$时，$\lambda_2=\frac {2}{13}$ ，此时$f(\lambda)=-\frac{2}{13}$。

##### 情况二：

当$\lambda_2=0$时，$\lambda_1=\frac {1}{4}$ ，此时$f(\lambda)=-\frac{1}{4}$。

因此我们确定最佳的参数选择为$\lambda_1=\lambda_3=\frac{1}{4},\lambda_2=0$

最后求解直线的参数：
$$
\begin{cases}
(\hat A,\hat B)=\frac{1}{4}\times1\times(3,3)+0\times(3,3)\times(4,3)-\frac{1}{4}\times1\times(1,1)=(\frac{1}{2},\frac{1}{2})\\
\hat C=1-(1\times\frac{1}{4}\times(3,3)·(3,3))-(1\times0\times(3,3)·(4,3))+(1\times \frac{1}{4}\times(3,3)·(1,1))=-2

\end{cases}
$$
得到我们的结果：
$$
l:\frac{1}{2}x+\frac{1}{2}y-2=0
$$
作图如图所示：

![image-20210509210859294](https://i.loli.net/2021/05/09/IecPsQUkxujJpgB.png)

## 三 · 更一般的情况

## 3.1 线性可分的数学定义

所谓高维度即指标增多的情况。在二维平面中，用于SVM划分分类的超平面是一条直线。但在更高维的情况中，线性可分有了更准确的定义：



> 对于 n 维欧氏空间中的两个点集![D_0](https://math.jianshu.com/math?formula=D_0)和![D_1](https://math.jianshu.com/math?formula=D_1)，若存在 n 维向量 ![w](https://math.jianshu.com/math?formula=w) 和实数 ![b](https://math.jianshu.com/math?formula=b)，使得：![\forall x_i\in D_0](https://math.jianshu.com/math?formula=%5Cforall%20x_i%5Cin%20D_0) 满足 ![w^Tx_i+b>0](https://math.jianshu.com/math?formula=w%5ETx_i%2Bb%3E0)，而 ![\forall x_j\in D_1](https://math.jianshu.com/math?formula=%5Cforall%20x_j%5Cin%20D_1) 满足 ![w^Tx_j+b<0](https://math.jianshu.com/math?formula=w%5ETx_j%2Bb%3C0)，则称![D_0](https://math.jianshu.com/math?formula=D_0)和![D_1](https://math.jianshu.com/math?formula=D_1)线性可分。在样本空间中，划分超平面可通过线性方程![w^Tx+b=0](https://math.jianshu.com/math?formula=w%5ETx%2Bb%3D0)来描述，其中![w=(w_1,w_2,...,w_d)](https://math.jianshu.com/math?formula=w%3D(w_1%2Cw_2%2C...%2Cw_d))为法向量，决定超平面的方向；![b](https://math.jianshu.com/math?formula=b)为位移项，决定超平面与原点之间的距离。一个划分超平面就由法向量![w](https://math.jianshu.com/math?formula=w)和位移![b](https://math.jianshu.com/math?formula=b)确定。



## 3.2 一般形式的线性可分SVM模型

一般情况下，我们面对的分类对象有多个指标，这些指标对应到空间中就成为了不同的维度。在如此高维的情况下，SVM的线性分类思想还是适用的，只需要在二维的空间上的模型进行修改即可。

### 3.2.1 $\gamma$的表示

与[1.2.1中的](#1.2.1$\gamma$的表示)$\gamma$类比，可得$\gamma$的表达式:
$$
\gamma=\frac{\hat\gamma}{||w||}
$$
其中$||w||$表示向量的二范式，即$||w||=\sqrt{w_1^2+w_2^2+...+w_p^2}$。

### 3.2.2 目标函数的改写

同类比可得：
$$
\begin {cases}
\min{\frac{1}{2}||w||^2}\\
s.t. y_i\times(wx_i+b)\geq1
\end {cases}
$$

### 3.2.3 问题的求解

同类比可得：
$$
\begin {cases}
\hat w=\sum_{i=1}^{n}\lambda_ly_ix_i\\
\hat b=y_j-\sum)_{i=1}^{n}\lambda_ly_i(x_i·y_j)
\end {cases}
$$

## 四 · 实际应用时的SVM模型优化

所谓优化，是根据使用情况以及使用情景对模型进行修改，从而让模型更好地工作。优化的方法有很多，这里我们通过现实情况进行一步步的探究与修改。

## 4.1 惩罚系数的引入

### 4.1.1 松弛因子$\xi$的引入

尽管我们的线性可分SVM在一些数据集中有较好的鲁棒性，但考虑现实中更可能出现的一种情况：

<img src="https://i.loli.net/2021/05/10/WXRydkfVop72TQg.png" alt="image-20210510100015234" style="zoom:50%;" />

在这种情况中，我们无法根据上述的规则确定一个超平面进行分类，仅用一条直线是无法将两类数据分离开的，因此我们可以考虑在**分类出现一定错误的情况下对**

**数据进行分类**：依然是用一个超平面将其分离，只不过这个超平面的两侧<font color='red'>允许有错误的分类点</font>，我们要做的是将错误的分类点尽可能少的出现。我们可以知道，在这种情况下不能找到一个可以将其分类的限制条件是这个约束条件：
$$
y_i(w^Tx_i+b)\geq1
$$
此时,任何的算法都无法给出这个约束条件下的解，因此我们需要改变约束条件。给定一个系数$\xi_i\geq0$(称之为松弛因子),将其的相反数加在不等式右边：
$$
y_i(w^Tx_i+b)\geq1-\xi_i\\即\\
y_i(w^Tx_i+b)+\xi_i \geq1\\
\forall i\in \{1,2,...,n\}
$$
这时候SVM的分离条件从将间隔设定为1改变成$1-\xi$，从而使得判定有了宽松的“谈判空间”。

### 4.1.2 惩罚系数C的引入

在4.1.1节中我们引入的$\xi$是大于零的实数，在根据需求设定时可以增大甚至增大至正无穷，但这时候的判定就会变得异常松弛，这是我们不希望见到的。人为选择参数是一个很复杂的过程，我们是否可以通过算法来确定松弛因子的最佳取值范围呢？在讨论松弛因子的时候我们研究了$y_i(w^Tx_i+b)$的约束条件，自然而然想要约束松弛因子的取值范围就需要考虑SVM的另一条件：
$$
\min\frac{1}{2}||w||^2+\sum_{i=1}^{n}\xi_i
$$
其中这个式子的前半部分无法修改，自然而然我们选择修改后半部分：
$$
\min\frac{1}{2}||w||^2+C\sum_{i=1}^{n}\xi_i
$$
其中$\C$称为惩罚系数，它会对$\xi$进行惩罚。具体的操作就是通过$\C$的设定影响间隔$\gamma$的大小，进而约束$\xi$。如果$C$取值很大，模型对误判惩罚力度也越大，因此模型必须通过牺牲间隔$\gamma$的宽度从而使得误判的犯错率减低，但会带来过拟合的问题；如果$C$取值较小，惩罚力度就越小，分类是否正确已经不是模型关心的重点，这样做导致模型失去了它的意义，也就是欠拟合的情况。

### 4.1.3 SVM的经典形式以及解

引入了松弛因子与惩罚系数后的方程称为经典形式的SVM方程。求解的过程与前面提到的一致，都是利用对偶性改写目标函数、求偏导数=0的方程解然后最后带入回原方程求解参数。

方程形式：
$$
\begin {cases}
\min{\frac{1}{2}||w||^2}+\sum_{i=1}^{n}\xi_i\\
s.t. y_i\times(wx_i+b)\geq1\\
\xi_i\geq0
\end {cases}
$$
方程的解：
$$
\begin {cases}
\hat w=\sum_{i=1}^{n}\lambda_ly_ix_i\\
\hat b=y_j-\sum)_{i=1}^{n}\lambda_ly_i(x_i·y_j)\\
0<\lambda_i<C
\end {cases}
$$

## 4.2 核函数（Kernels）的引入

### 4.2.1 一个棘手的问题

利用`python`的`sklearn`库的`make_circle`函数生成一段数据：

```python
x1,y1=make_circles(n_samples=1000,factor=0.5,noise=0.1)#x1是坐标数组，y1是类别，生成一千个样本，AB类平均间隔为0.5，噪声干扰占0.1
#画图函数过长忽略
```

![image-20210510154022608](https://i.loli.net/2021/05/10/4ki8lByTnHIJXUw.png)

在这个问题中，尽管我们优化了线性SVM的模型，但从散点图的情况来看似乎是无解的。难道花了大力气推导出来的SVM只能在有限的情况下发挥它的功能吗？其实我们需要换个角度来思考。在高等数学中我们在求一些形式复杂的积分时，会选择换元，其中有一种换元叫做坐标轴变换。坐标系的变换，从常见的直角坐标系，到极坐标系，到后来的柱坐标、球坐标都有。在这个情况中，我们是否也可以将数据的坐标表示进行转换呢？

### 4.2.2 空间的变换

不妨定义一个映射$\Phi:X -> \Z$,它能将原本的坐标空间映射到另一个坐标空间。在之前的讨论中，我们的空间X是基于实数的n维空间$X=\R^d$，我们可以将它的映射关系写为：
$$
\Phi(x)=[x^1,x^2,...,x^n]^T
$$
它其实是一种最基本的核函数，称为**线性核**。如果将维度进行扩展，比如说二维扩展$X=\R^{d^2}$,那么映射关系可以变成：
$$
\Phi(x)=[x^1,x^2,...,x^n，x^1 x^1,..x^1x^n,...x^nx^1,...x^nx^n]^T
$$

当我们的维度扩展至无限时，称为高斯核函数：

除此之外还有其他的核函数。核函数是利用SVM解决问题的关键，SVM的成功与否很大程度上取决于核函数的设计。

### 4.2.3 核函数的使用

核函数是本质是一种坐标的映射，因此将其带入进与坐标相关的位置即可：
$$
\begin {cases}
min\frac{1}{2}\sum_{i}^{n}\sum_{j}^{n}\lambda_i\lambda_jy_iy_jK(x_i,x_j)-\sum_{i}^{n}\lambda_i\\
s.t. \sum_i^n\lambda_iy_i=0,\lambda_i\in[0,C]
\end{cases}
$$
在这个例子中，我们可以将二维的数据进行拓展，将其拓展至三维，这是我们要通过现有的两个指标来确定第三个指标。首先必须引入与现有两个指标无关的参考量从而使得到的新指标与前两个无关，简单来说从由三个未知数组成的方程若有解则它们方程对应的矩阵一定要线性无关才行。我们可以引入一个新的指标：点到原点的距离：
$$
z_i=\sqrt{x_1^2+x_2^2+...+x_n^2}
$$
在这个例子中n取2即可。这样我们将原有的坐标数据$(x_i,y_i)$改写成$(x_i,y_i,z_i)$。从而完成坐标的映射，因此，我们的SVM分类可以继续了：

![image-20210510172035096](https://i.loli.net/2021/05/10/zrXLNyqhS8Bt2CE.png)

在这种情况下求解超平面变得可行。如果对分类的结果不满意的话，可以使用其他的核函数以获得更好的拟合。



## 五 · 总结

SVM是一种广泛应用在于解决模式识别领域中的数据分类问题，属于有监督学习算法的一种。其算法的核心是“最大化间隔”，在本文仅探讨了线性的情况以及非线性转换成近似线性的一种思路。代码分析可见后续。