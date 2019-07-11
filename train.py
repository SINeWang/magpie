from magpie import Magpie

magpie = Magpie()
magpie.train_word2vec('data/hep-categories', vec_dim=3) #训练一个word2vec
magpie.fit_scaler('data/hep-categories') #生成scaler
magpie.init_word_vectors('data/hep-categories', vec_dim=3) #初始化词向量
labels = ['军事','旅游','政治'] #定义所有类别
magpie.train('data/hep-categories', labels, test_ratio=0.2, epochs=20) #训练，20%数据作为测试数据，5轮

#保存训练后的模型文件
magpie.save_word2vec_model('save/embeddings/here', overwrite=True)
magpie.save_scaler('save/scaler/here', overwrite=True)
magpie.save_model('save/model/here.h5')