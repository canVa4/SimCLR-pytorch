## 写在开头

整体的代码使用pytorch实现，基于https://github.com/sthalles/SimCLR （用pytorch实现simCLR中star最多的）实现了Logistic Loss（支持使用欠采样、改变权重和无操作）和margin triplet loss（支持semi-hard mining），并可选LARS（experimental）和ADAM优化。代码框架支持resnet50和resnet18；dataset支持STL10和CIARF10（测试时使用CIARF10）

训练为：*run.py*；修改训练参数只需要修改：*config.yaml* ；评估使用*evluation.py*（分开的原因是因为我租了GPU，用GPU训练，用我的PC测试，这样可以更快一些）

个人运行环境：win10 + pytorch 1.5 + cuda 10.2（租的GPU 1080ti）

| 日期                      | 进度                                                         |
| ------------------------- | ------------------------------------------------------------ |
| 5-19 Tue（基本满课+实验） | 论文阅读，选定使用pytorch实现和决定基于上文链接实现代码      |
| 5-20 Wed                  | 熟悉基础知识、了解代码整体框架，理解loss function，并进行初步尝试编写loss，未调试 |
| 5-21 Thu（满课+实验）     | 写完了evaluation部分                                         |
| 5-22 Fri（基本满课）      | 跑代码，发现只用CPU究极龟速；于是装cuda，结果装了一白天的cuda T.T，晚上测试代码并初步验证loss function是否书写正确；初步移植LARS |
| 5-23 Sat                  | 测试三个Loss并尝试调参，尝试使用resnet18作为backbone网络，旁晚开始租了个GPU来跑模型，实现triplet loss(sh) |
| 5-24 Sun                  | 调参、修复bug、跑代码、微调loss（Logistic loss增加欠采样和改变权重） |
| 5-25 Mon                  | 调参、跑代码                                                 |

## Results

Linear evaluation均使用Logistic Regression，均train from scratch（no pretrain）

GPU: 1080ti    resnet50训练+测试一次需5.5h；resnet18训练+测试一次需2.6h

| batch | epoch | out dim | optimizer | Loss                   | BackBone | t/m  | CIARF10 Top-1 |
| ----- | ----- | ------- | --------- | ---------------------- | -------- | ---- | ------------- |
| 128   | 80    | 128     | ADAM      | NT-Xent                | resnet50 | 0.1  | 78.1%         |
| 128   | 80    | 128     | ADAM      | NT-xent                | resnet50 | 0.5  | 79.3%         |
| 128   | 80    | 128     | ADAM      | NT-Xent                | resnet50 | 1    | 77.2%         |
| 128   | 80    | 128     | ADAM      | Triplet Loss           | resnet50 | 0.4  | 65.1%         |
| 128   | 80    | 128     | ADAM      | Triplet Loss           | resnet50 | 0.8  | 70.7%         |
| 128   | 80    | 128     | ADAM      | Triplet Loss(sh)       | resnet50 | 0.8  | 73.5%         |
| 128   | 80    | 128     | ADAM      | NT-Logistic(none)      | resnet50 | 0.2  | 37.5%         |
| 128   | 80    | 128     | ADAM      | NT-Logistic (sampling) | resnet50 | 0.2  | 62.4%         |
| 128   | 80    | 128     | ADAM      | NT-Logistic (sampling) | resnet50 | 0.5  | 69.9%         |
| 128   | 80    | 128     | ADAM      | NT-Logistic (sampling) | resnet50 | 1    | 66.2%         |
| 128   | 80    | 128     | LARS      | NT-xent                | resnet50 | 0.5  | TODO          |
| 128   | 80    | 128     | ADAM      | NT-xent                | resnet18 | 0.5  | 72.4%         |
| 128   | 80    | 128     | ADAM      | NT-Logistic(weight)    | resnet18 | 0.2  |               |

## 损失函数

对于每一个输入图片，模型会生成两个representation，最终优化的目标可以理解为：同一个batch内来自同一张图片的两个representation的距离近，让来自不同输入图片的representation的距离远。注意，论文中给出的是negative loss function

### Logistic Loss

首先给出论文中的形式（negative loss function）：

* $$ log \sigma(u^Tv^+/\tau) + log\sigma(-u^T v^-/ \tau) $$

这里对于此公式，我一开始是没有理解的，于是自己尝试推理了一下。

对于每一个输入样本，模型会生成两个representation，对于一个有N个输入的batch的，就会产生2*N个representation，对于每一对representation计算一个cosine similarity。而每一对representation（下文用 $(i,j)$ 序偶来表示他们）可以根据他们的来源来确定他们label（即：来自同一输入的为正类，来自不同输入的为反类），这样就构成了一个监督任务。

将这个任务看为监督后，因为论文中提到的这个损失函数的名字是logistic loss，我自然地想到了logistics regression。于是从这个角度入手，来推理这个loss function。

用$ P(i,j) $表示一对representation为正类的概率。设正类y=1，反类y=0

那么写出整个数据集的对数似然函数$$ LL(\theta;X)=\sum_{each(i,j)} (y_{(i,j)} logP(i,j)+(1-y_{(i,j)})log(1-P(i,j)) )$$

对上式化简可以得到：$$ LL(\theta;X)=\sum_{正类} logP(i,j)+\sum_{反类}log(1-P(i,j)) $$

而cosine similarity并不是一个[0,1]之间的数（或者说没有概率的意义），参照logistics regression，将cosine similarity经过一个sigmoid函数$$ \sigma( \cdot) $$ 之后就变为了一个[0,1之间的数]，而且对于sigmoid有$$ \sigma(-x)=1-\sigma(x) $$,所以有：$$ LL(\theta;X)=\sum_{正类} log[\sigma(sim(i,j))]+\sum_{反类}log[\sigma(-sim(i,j))] , sim(i,j)为(i,j)的相似度指标$$

只需引入temperature就可将上式变为与论文中公式相同的形式。

在使用原版loss时，发现最终结果效果很差（见result中的NT-Logistics none）。个人猜测原因如下：

* 样本非常不均衡，正例对远远少于反例。

解决办法：

* 对反例样本对使用简单的under-sampling（欠采样）
* 对于loss计算时，正反例样本设置不同的权重

（注：由于训练时间太久，没有来得多次跑weight测试效果）

代码实现：

``` python
    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)
        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1) * -1

        logits_pos = self.sigmoid(positives / self.temperature).log_()
        logits_neg = self.sigmoid(negatives / self.temperature).log_()
        if self.method == 1:
            # under-sampling
            all_one_vec = np.ones((1, 2 * self.batch_size,))
            all_zero_vec = np.zeros((1, 2 * self.batch_size * (2 * self.batch_size - 3)))
            under_sampling_matrix = np.column_stack((all_one_vec, all_zero_vec)).flatten()
            np.random.shuffle(under_sampling_matrix)
            under_sampling_matrix = torch.tensor(under_sampling_matrix).view(
                (2 * self.batch_size, 2 * self.batch_size - 2)).type(torch.bool).to(self.device)

            logits_neg = logits_neg[under_sampling_matrix]
            loss = torch.sum(logits_pos) + torch.sum(logits_neg)
            return -loss
        elif self.method == 2:
            # change weight
            neg_count = 2*self.batch_size*(2*self.batch_size - 2)
            pos_count = 2*self.batch_size
            loss = neg_count * torch.sum(logits_pos) + pos_count*torch.sum(logits_neg)
            return -loss/(pos_count+neg_count)
        else:
            # none
            total_logits = torch.cat((logits_pos, logits_neg), dim=1)
            loss = torch.sum(total_logits)
            return -loss

```

### Margin Triplet

首先给出论文中的形式（negative loss function）：

* $$ -max(u^Tv^--u^Tv^+m,0)$$

此公式理解起来相对直观，即对于一个输入样本，计算其和一个负样本相似度减去和正样本的相似度在加上m，并与0取max。该m可以理解：m越大为希望正反样本分开的距离越大。其目标是希望输入样本和正样本的相似度减去和负样本的相似度可以大于阈值m值。下图很形象的描述了这些关系。

<img src="D:\MLandDeeplearning\SimCLR\SimCLR-pytorch\image\triplet1.PNG" style="zoom:67%;" />

所以，对于每一个输入样本k，该样本的**margin tripl loss**为$$ \sum_{i}^{所有反类}max(u_k^Tv_i^--u_k^Tv^+m,0) $$

所以总的loss就是将所有输入样本的loss加起来

* $$ \frac{1}{2N*(2N-2)}\sum_{k}^{所有正类}\sum_{i}^{所有反类}max(u_k^Tv_i^--u_k^Tv^++m,0) $$

* 同时也实现了semi-hard negative mining. 即计算loss（梯度）时，只考虑上图中semi-hard negatives的loss。即选择满足：$$ u^Tv^++m>u^Tv^-$$

代码实现：

```python
       def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)
        similarity_matrix = self.similarity_function(representations, representations)
        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        mid = similarity_matrix[self.mask_samples_from_same_repr]
        negatives = mid.view(2 * self.batch_size, -1)
        zero = torch.zeros(1).to(self.device)
        triplet_matrix = torch.max(zero, negatives - positives + self.m_param)
        # max( sim(neg) - sim(pos) + m, 0)
        # 2N,2N-2 每一行代表了对于一个z关于其正类（z+batch）和其他反类的triplet loss
        if self.semi_hard == True:
            # semi-hard
            semi_hard = - negatives + positives + self.m_param
            # print(semi_hard)
            semi_hard_mask = torch.max(semi_hard, zero).type(torch.bool)
            # print(semi_hard_mask)
            triplet_matrix_sh = triplet_matrix[semi_hard_mask]
            # print(triplet_matrix)
            # print(triplet_matrix_sh)
            loss = torch.sum(triplet_matrix_sh)
            return loss
        else:	# normal
            loss = torch.sum(triplet_matrix)     
            return loss / (2*self.batch_size*(2*self.batch_size - 2))
```

### NT-Xent

论文中的形式：

* $$ l(i,j)=-log \frac{exp(s_{i,j}/\tau)}{\sum^{2N}_{k=1} 1_{k\not=i}exp(s_{i,j}/\tau)}$$, $$ L = \frac{1}{2N} \sum^{N}_{k=1}[l(2k-1,2k)+l(2k,2k-1)]$$

代码实现未进行修改。

## simCLR模型

主要使用ResNet-50来实现，参照论文B.9中所写：将Resnet第一个卷积层改为了3*3的Conv，stride=1，并去除第一个max pooling层；在augmentation中去除了Guassian Blur。

projection head同论文中一样，使用两层的MLP。

## 遇到的问题与解决方法

Q1：使用个人笔记本训练，显存不足，使用cpu训练耗时过久。

A1：尝试使用过resnet18，仍时间仍很长，最终决定租GPU（1080ti）来训练。

Q2：训练时发现最终测试结果不好。

A2：最终控制变量，与未修改的代码对比测试，发现个人版本在sampler的时候不小心去掉了很多的训练样例，已修复为同原版。修复后，基本同原版效果

Q3：使用LARS效果不好，loss不能稳定下降，震荡严重。（unsolve）

A3：尝试修改debug，修改学习率，由于时间不足，暂未解决。

## 关于Loss的个人想法

从测试结果和论文结果可以看出，NT-xent的效果更佳。个人认为其主要的优势在于：

* NT-xent（cross entropy）利用的是相对相似度，而其余二者不是。这样可以缓解个别样本差异过大导致的不均衡（感觉类似于input的normalization）。
* NT-xent计算了所有positive pair的loss。而NT-logistic和Margin Triplet则使用全部的pair来计算，不使用semi-hard mining的话，可能会造成坍塌。对于此模型生成的样本，可以看到其样本类别并不均衡，对于NT-logistic，这可能会导致训练效果下降。（使用semi-hard negative mining、采样、改变权重可以缓解这个问题）

经过自己的implement之后，实在是羡慕google的TPU集群了！

这是我第一次真正接触self-supervised learning，之前只是有所耳闻，感觉这种contrastive learning的想法真的很有趣。