# coding:utf-8

class Config(object):
    init_scale = 0.04                           #均勻分布的上下界
    learning_rate = 0.001                       #學習率
    max_grad_norm = 15                          #對梯度進行規範
    num_layers = 3                              #RNN的層級數
    num_steps = 25                              #每次訓練多少字
    hidden_size = 1000                          #神經網路隱含層的維度
    iteration = 40                              #模型迭代次數

    save_freq = 5                               #每迭代多少次保存一次模型
    keep_prob = 0.5                             #dropout的概率
    batch_size = 32                             #min-batch的大小
    model_path = './model/Model'                #model路徑
  
    save_time = 40                              #載入第幾次保存的模型
    is_sample = True                            #是否使用sample，若False則是max
    is_beams = True                             #是否使用beam-search解碼
    beam_size = 2                               #beam-search的窗口大小
    len_of_generation = 50                      #生成字數
    start_sentence = u'他'                      #開始句子
