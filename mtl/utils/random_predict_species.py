import pandas as pd
import numpy as np

def random_predict(nb_pred, traindata_path='/home/wangbin/mtl/data/pfam/train_data_embedding.pkl', species_path='/home/wangbin/mtl/data/pfam/species.pkl'):
    df_train = pd.read_pickle(traindata_path)
    labels = list(pd.read_pickle(species_path).species.values)
    train_data = list(df_train.species.values)
    # 统计每个标签出现的次数和概率
    label_count = {label: 0 for label in labels}
    for data in train_data:
        for label in data:
            label_count[label] += 1

    label_prob = {label: count / len(train_data) for label, count in label_count.items()}
    # 随机预测
    nb_pred = 10
    random_prob = np.random.uniform(size=(nb_pred, len(labels)))
    result = np.where(random_prob < np.array(list(label_prob.values())), 1, 0)

    # 打印结果
    predict_result = list()
    for i in range(nb_pred):
        predict_result.append([label for label,prob in zip(labels,result[i]) if prob == 1])
        # print(f"Sample {i}: {dict(zip(labels, result[i]))}")

    return predict_result

