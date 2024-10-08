import pandas as pd
import numpy as np
import tensorflow as tf
from Config import Config
import os
import Hyperparameters
import sys
def get_bitrate(state):
    current_directory = str(os.path.dirname(os.path.realpath(__file__)))
    config = Config()
    N = config.N

    tile_num = config.tile_num  # 切块数量
    level_num = config.level_num  # 质量等级数量
    x_num = config.x_num
    s_len = config.s_len
    class QNetwork(tf.keras.Model):
        @tf.function
        def __init__(self):
            super().__init__()
            self.dense1 = tf.keras.layers.Dense(units=s_len, activation=tf.keras.layers.LeakyReLU(alpha=0.02))
            self.drop1 = tf.keras.layers.Dropout(0.05)
            self.bn1 = tf.keras.layers.LayerNormalization(axis=-1)
            self.dense2 = tf.keras.layers.Dense(units=s_len, activation=tf.keras.layers.LeakyReLU(alpha=0.02))
            self.drop2 = tf.keras.layers.Dropout(0.05)
            self.bn2 = tf.keras.layers.LayerNormalization(axis=-1)

            self.dense3 = tf.keras.layers.Dense(units=s_len, activation=tf.keras.layers.LeakyReLU(alpha=0.02))
            self.drop3 = tf.keras.layers.Dropout(0.05)
            self.bn3 = tf.keras.layers.LayerNormalization(axis=-1)
            self.V_dence = tf.keras.layers.Dense(units=1, activation=tf.keras.layers.LeakyReLU(alpha=0.02))

            self.A_dence = tf.keras.layers.Dense(units=x_num, activation=tf.keras.layers.LeakyReLU(alpha=0.02))


        @tf.function
        def call(self, inputs):         #输出Q值
            FN = self.dense1(inputs)
            FN = self.drop1(FN)
            FN = self.bn1(FN)
            FN = self.dense2(FN)
            FN = self.drop2(FN)
            FN = self.bn2(FN)
            FN = self.dense3(FN)
            FN = self.drop3(FN)
            FN = self.bn3(FN)
            svalue = self.V_dence(FN)
            avalue = self.A_dence(FN)
            mean = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x,axis=1,keepdims=True))(avalue)   #用Lambda层，计算avg(a)
            advantage = avalue - mean             #a - avg(a)

            output = svalue + advantage

            x = output
            return x
        @tf.function
        def posibility(self,input):
            x = self.call(input)
            x = tf.nn.softmax(x)
            return x

        @tf.function
        def predict(self, inputs):      #输出行动
            q_values = self(inputs)

            return tf.argmax(q_values, axis=-1)
    model = QNetwork()
    model.load_weights(current_directory+'/DDQN_model/DDQN_model_variables/mymodel1995')
    model(tf.constant([state]))
    result=model(tf.constant([state])).numpy()[0]
    ret=[0]*tile_num
    for i in range(tile_num):
        cur_prob=-9999
        for j in range(level_num):
            if result[i*level_num+j]>cur_prob:
                ret[i]=j
                cur_prob=result[i*level_num+j]
    return ret
