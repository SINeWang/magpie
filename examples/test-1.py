import os
import sys
sys.path.append(os.path.realpath(os.getcwd()))
sys.path.append("..")

from magpie import Magpie

magpie = Magpie(
    keras_model='../workspace/model.h5',
    word2vec_model='../workspace/embeddings',
    scaler='../workspace/scaler',
    labels=['旅游', '军事', '政治']
)

# 单条模拟测试数据
text = '特朗普在联合国大会发表演讲谈到这届美国政府成绩时，称他已经取得了美国历史上几乎最大的成就。随后大会现场传出了嘲笑声，特朗普立即回应道：“这是真的。”'
mag1 = magpie.predict_from_text(text)
print(mag1)

'''
#也可以通过从txt文件中读取测试数据进行批量测试
mag2 = magpie.predict_from_file('data/hep-categories/1002413.txt')
print(mag2)
'''
