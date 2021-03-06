# chinese_asr

|作者|艾伦爱干饭|
|----|----|
|爱好|干饭|
|公众号|待更|

## 目录
* [项目背景](#项目背景)
* [功能特性](#功能特性)
    * [预处理](#预处理)
    * [ctc-loss](#ctc-loss)
    * [解码](#解码)
* [环境依赖以及使用](#环境依赖以及使用)
* [模型下载](#模型下载)
* [数据集下载](#数据集下载)
* [计划](#计划)


项目背景
------
实习后对语音识别很有兴趣，加上网上资料较少，因此动手做了一个端到端的中文语音识别模型训练框架

功能特性
------
本asr框架功能包括数据预处理、模型训练、解码、计算wer

### 预处理
- 提取80维fbank特征
- 标准化：计算特征的均值和方差进行标准化，加快模型收敛速度，因为计算标准化的速度较慢，已将参数放到data_file/thchs_30/stand_nor.txt文件中，可直接调用

### ctc-loss
- 一段音频，提取到的特征的数量是大于label中词的数量的，这是一个多对少的问题，而ctc-loss通过加入blank，能够解决时序类数据分类时的对齐问题
### 解码
- 采用greedy search

环境依赖以及使用
------
- 下载thch_30数据集后放到data
- 根据requirement安装依赖安装包
- 在linux系统下输入以下代码，进行训练
- python --pattern train --decode greedy
- 在linux系统下输入以下代码，进行测试
- python --pattern test --decode greedy

模型下载
------
|模型|数据集|训练wer|验证wer|链接|备注|
|----|----|----|----|----|----|
|bi_lstm_150_c2|thchs_30|6.7%|18.66%|https://pan.baidu.com/s/1VVavLKLeY584HudHtC5uIQ 提取码：n18y|2层bi-lstm+ctc|

数据集下载
-----
thchs_30 下载地址：http://www.openslr.org/18/

### 计划

- 更新tranformer，conformer模型
- 更新其他数据集的训练方法
- 更新其他解码方法
