# 天池—真实场景篡改图像检测挑战赛

队伍名称：欢乐摸鱼~

团队成员：今天也要摸鱼呀~、liyingxuan、瑶瑶子可可爱爱

## 环境

系统：Ubuntu 20.04

显卡：2080Ti

## Test

代码结构和环境配置方式按照天池复赛的代码规范和docker要求进行整理。

- 可以直接拉取我上传到阿里云的镜像（公有仓库）

  registry.cn-hangzhou.aliyuncs.com/liyingxuan/tianchi_lyx:final

- 如果重新build镜像的话


```bash
docker build -t tianchi .
```

运行（要挂载一下存储）：

```bash
nvidia docker run tianchi -v <path>
```

单模型推理：

```bash
python code/inference.py --model unet --pretrained-ckpt <pretrained-path>
```

多模型推理（需要自己去py文件中修改一下路径）

```bash
python code/inference_model_ensemble.py 
```

## Train

具体的方案见“方案说明”

```bash
cd code
python train.py --model unet --work-dir ./work_dir/unet/
```

## 方案说明

按照五折交叉验证的方式对训练集进行划分

### baseline

- 尝试了mvssnet，rrunet，最终选择了以efficientnet-b5为encoder的unet++网络
- 损失函数为diceloss和bceloss
- 图片resize成512*512大小
- AdamW优化器
- 对训练集的4000张数据按照五折交叉验证的方式进行划分，训练过程中保存验证集上得分最高的模型

初赛时，Unet单模最高分数大约为2350分，mvssnet大约1900分，rrunet大约2100分。将encoder换成efficientnet-b7会更好，但是我2080Ti太小了，训练效率很低。

### 滑窗裁剪

- 考虑到对图片进行缩放会丢失太多的细节信息，不利于篡改区域检测，所以使用滑动窗口的方法将图像裁剪成512*512的小块，重叠区域为128
- 推理的时候也将图像进行裁剪，重叠区域的预测结果取平均
- 但有时候需要结合整张图片的全局信息才比较好进行判断，将图像resize成512分辨率输入网络得到的结果记为$M_{resize}$，将滑窗裁剪得到的结果记为$M_{slice}$，最终预测结果为$\alpha M_{resize} + (1-\alpha)M_{slice}$
- 调整阈值$th$（在验证集上设置不同的阈值，进行搜索得到最优解，$\alpha$也是在验证集上搜索得到），搜索的代码为`th-search.py`

仅仅用2350分的模型（直接resize图片进行训练），但在推理时采用$0.7M_{resize} + 0.3M_{slice}$的策略，阈值设为0.3，单模分数就达到了2560。

后来对训练集也进行裁剪，推理时$M = 0.8M_{resize} + 0.2M_{slice}$，阈值设为0.4，单模分数差不多是2700。

### 数据增强

- 加入了sea3的训练数据（效果其实很小，在baseline的基础上单模大概提升了40-50分）
- 对图像进行二次篡改，包括从同一张图像随机复制一块，从另一张图像随机复制一块，以及随机擦除，离线生成了3000张二次篡改图像（这里感谢Dave大佬开源的方案）
- 或者在线进行数据增强（`dataset.py`中的`ManiDatasetAug`类）

数据增强用离线和在线效果差不多，大约提升了30-40分。

### 模型集成

- 2-3个模型的预测结果取平均再进行阈值化

一般能在单模的基础上提升50-150分不等，有时候又会起到反效果，看脸。

来不及训练5个模型，如果可以的话可能提升更多。

### 半监督

用提交分数最高的模型给测试集打伪标签，借鉴Tri-training的方式筛选部分测试集加入训练集中，具体为：

- 用fold0，fold1和fold2训练的模型model0，model1和model2分别推理测试集的样本，得到的结果pre0，pre1和pre2，并经过阈值化得到res0，res1和res2
- 对于model0，计算res1和res2中交并比最高的$x$张图像，并将对应的pre1和pre2的结果进行融合，并二值化，加入model0的新训练集中
- model1和model2执行类似model0的操作，筛选$x$张测试集及其伪标签加入训练集中
- 迭代，并在迭代的过程中逐渐增大$x$，我是500、1000、1500、2000、2500

分数提升很明显，初赛提升200多分（应该没到极限），复赛提升100多。但是$x$增大到3000的时候分数就开始下降了，说明伪标签的质量并不高。

### 最终方案

- efficientnet-b5，Unet
- 加入sea3的训练数据
- 生成二次篡改图像3000张 or 在线数据增强
- 半监督，打伪标签，迭代训练
- 集成了两个模型：使用768分辨率训练的模型（resize），使用512滑窗切片训练的模型（$0.8M_{resize} + 0.2M_{slice}$）

### 分数：

初赛：2961/3000张

复赛：1440/2000张

### 不足

- 前期对语义分割任务了解不多，忽视了backbone的重要性，从前排大佬的讨论来看用swin Unet或者ConvNeXt-XL作为baseline单模的分数就会很高。从复赛top1和top5大佬的方案来看，他们都是使用了比较大的模型作为baseline，所以初赛阶段不加任何技巧单模的分数就能到2900分。
- 可以尝试更为先进的半监督方案，top1大佬的方式是，对于预测结果中大于0.7的，认为是篡改区域，小于0.3的认为是非篡改区域，0.3-0.7的认为是置信度较低的区域，生成一个mask，计算loss的时候忽略这部分区域，使用所有4000张的测试集数据进行半监督训练。

## TOP方案复盘

什么是真正的暴力美学。

因为天池不强制要求选手公开自己的方案，我目前只看到了TOP1和TOP5的方案。

TOP1和TOP5都使用了swin-v2或ConvNeXt-L这种大模型，然后将分辨率调整到1280，加上半监督。

## 参考资料

- [关于数据增强-天池技术圈-天池技术讨论区 (aliyun.com)](https://tianchi.aliyun.com/forum/postDetail?spm=5176.12586969.1002.3.5de127f0Rf2Aep&postId=348449)
- [qubvel/segmentation_models.pytorch: Segmentation models with pretrained backbones. PyTorch. (github.com)](https://github.com/qubvel/segmentation_models.pytorch)

## 致谢

最后，感谢主办方举办此次比赛，为我们提供了锻炼的平台，还在群里耐心解答我们的疑惑。感谢大佬开源的方案，让没接触过这类任务的我们也能很快上手。感谢天池提供的交流和学习的机会，让我结识了非常靠谱的队友，可以一起学习，共同进步。
