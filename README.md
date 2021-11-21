# chinese_asr

|作者|艾伦爱干饭|
|----|----|
|爱好|干饭|
|公众号|待更|

## 目录
* [项目背景](#项目背景)
* [功能特性](#文本)
* [环境依赖以及安装]
* [环境依赖以及安装]
* [环境依赖以及安装]
* [环境依赖以及安装]
### 项目介绍
实习后对语音识别很有兴趣，加上网上资料较少，因此动手做了一个端到端的中文语音识别模型训练框架
### 功能特性
本asr框架功能包括数据预处理、模型训练、解码、计算wer

### 环境依赖
### 训练数据
训练数据采用thchs_30，下载地址：http://www.openslr.org/18/
下载后将data_thchs30解压到data文件中
### 使用模型
bi-lstm  + ctc
可直接将语音转换为中文

## 生成训练文件格式
![image](https://user-images.githubusercontent.com/53568883/142419223-2640cd2c-8479-4a92-b977-798eb5136298.png)

## 字典
_ 作为ctc的blank

![image](https://user-images.githubusercontent.com/53568883/142418123-b8314cbc-c091-493e-a394-9eb59175c44c.png)

### 预处理
1、提取80维fbank特征

2、标准化：计算特征的均值和方差进行标准化，加快模型收敛速度，因为计算标准化的速度较慢，已将参数放到data_file/thchs_30/stand_nor.txt文件中，可直接调用

### 模型下载

bi_lstm_150_nl， 测试集wer为29.278% 

百度云链接：https://pan.baidu.com/s/1VVavLKLeY584HudHtC5uIQ  

提取码：n18y

### 安装和使用

### 计划
1、加入CNN降维度

2、更新tranformer，conformer模型

3、更新其他数据集的训练方法

4、更新其他解码方法
