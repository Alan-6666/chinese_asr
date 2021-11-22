# chinese_asr

|作者|艾伦爱干饭|
|----|----|
|爱好|干饭|
|公众号|待更|

## 目录
* [项目背景](#项目背景)
* [功能特性](#功能特性)
    * [预处理](#预处理)
    * [解码](#解码)
    * [计算wer](#计算wer)
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
### 解码
- 采用greedy search

环境依赖以及使用
------
- 下载thch_30数据集后放到data
- 根据requirement
模型下载
------
|模型|数据集|训练wer|测试wer|链接|备注|
|----|----|----|----|----|----|
|bi_lstm_150_nl|thchs_30|12%|29.278%|https://pan.baidu.com/s/1VVavLKLeY584HudHtC5uIQ 提取码：n18y|一层双向LSTM+ctc|

数据集下载
-----
下载地址：http://www.openslr.org/18/

### 计划

- 更新tranformer，conformer模型
- 更新其他数据集的训练方法
- 更新其他解码方法
