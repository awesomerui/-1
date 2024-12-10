import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import MDS, LocallyLinearEmbedding, Isomap
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV

# 加载LFW数据集
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
data = lfw_people.data  # 图像数据
label = lfw_people.target  # 图像对应的标签
n_classes = len(np.unique(label))  # 目标类别的数量

# 数据分离
X_train, X_test, y_train, y_test = train_test_split(
    data, label, test_size=0.25, random_state=42
)  # 将数据集分为训练集和测试集，25%的数据作为测试集

# 降维维数
n_components = 150  # 降维后的特征数量

# 高斯贝叶斯分类器
gnb = GaussianNB()

# 存储结果
y_scores = {}  # 存储每种降维方法下的决策函数得分
y_accuracies = {}  # 存储每种降维方法下的准确率

# PCA降维
pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# 特征选择
lasso = LassoCV(cv=5)  # 使用交叉验证来选择最优的Lasso正则化参数
lasso.fit(X_train_pca, y_train)
selector = SelectFromModel(lasso, prefit=True)  # 根据Lasso模型的系数选择重要特征
X_train_selected_pca = selector.transform(X_train_pca)
X_test_selected_pca = selector.transform(X_test_pca)

gnb.fit(X_train_selected_pca, y_train)  # 使用特征选择后的训练数据训练贝叶斯分类器
y_pred_pca = gnb.predict(X_test_selected_pca)  # 进行预测
y_accuracy_pca = accuracy_score(y_test, y_pred_pca)  # 计算准确率
print(f"PCA Accuracy: {y_accuracy_pca}")

y_accuracies['PCA'] = y_accuracy_pca  # 存储准确率

# KPCA降维
kpca = KernelPCA(n_components=n_components, kernel='cosine').fit(X_train)
X_train_kpca = kpca.transform(X_train)
X_test_kpca = kpca.transform(X_test)

# 特征选择
lasso.fit(X_train_kpca, y_train)
selector = SelectFromModel(lasso, prefit=True)
X_train_selected_kpca = selector.transform(X_train_kpca)
X_test_selected_kpca = selector.transform(X_test_kpca)

gnb.fit(X_train_selected_kpca, y_train)  # 使用特征选择后的训练数据训练贝叶斯分类器
y_pred_kpca = gnb.predict(X_test_selected_kpca)  # 进行预测
y_accuracy_kpca = accuracy_score(y_test, y_pred_kpca)  # 计算准确率
print(f"KPCA Accuracy: {y_accuracy_kpca}")

y_accuracies['KPCA'] = y_accuracy_kpca  # 存储准确率

# MDS降维
mds = MDS(n_components=n_components).fit(X_train)
X_train_mds = mds.fit_transform(X_train)
X_test_mds = mds.fit_transform(X_test)

# 特征选择
lasso.fit(X_train_mds, y_train)
selector = SelectFromModel(lasso, prefit=True)
X_train_selected_mds = selector.transform(X_train_mds)
X_test_selected_mds = selector.transform(X_test_mds)

gnb.fit(X_train_selected_mds, y_train)  # 使用特征选择后的训练数据训练贝叶斯分类器
y_pred_mds = gnb.predict(X_test_selected_mds)  # 进行预测
y_accuracy_mds = accuracy_score(y_test, y_pred_mds)  # 计算准确率
print(f"MDS Accuracy: {y_accuracy_mds}")

y_accuracies['MDS'] = y_accuracy_mds  # 存储准确率

# LLE降维
lle = LocallyLinearEmbedding(n_components=n_components, n_neighbors=200, neighbors_algorithm='auto').fit(X_train)
X_train_lle = lle.transform(X_train)
X_test_lle = lle.transform(X_test)

# 特征选择
lasso.fit(X_train_lle, y_train)
selector = SelectFromModel(lasso, prefit=True)
X_train_selected_lle = selector.transform(X_train_lle)
X_test_selected_lle = selector.transform(X_test_lle)

gnb.fit(X_train_selected_lle, y_train)  # 使用特征选择后的训练数据训练贝叶斯分类器
y_pred_lle = gnb.predict(X_test_selected_lle)  # 进行预测
y_accuracy_lle = accuracy_score(y_test, y_pred_lle)  # 计算准确率
print(f"LLE Accuracy: {y_accuracy_lle}")

y_accuracies['LLE'] = y_accuracy_lle  # 存储准确率

# ISOMAP降维
isomap = Isomap(n_components=n_components, n_neighbors=200, neighbors_algorithm='auto').fit(X_train)
X_train_isomap = isomap.transform(X_train)
X_test_isomap = isomap.transform(X_test)

# 特征选择
lasso.fit(X_train_isomap, y_train)
selector = SelectFromModel(lasso, prefit=True)
X_train_selected_isomap = selector.transform(X_train_isomap)
X_test_selected_isomap = selector.transform(X_test_isomap)

gnb.fit(X_train_selected_isomap, y_train)  # 使用特征选择后的训练数据训练贝叶斯分类器
y_pred_isomap = gnb.predict(X_test_selected_isomap)  # 进行预测
y_accuracy_isomap = accuracy_score(y_test, y_pred_isomap)  # 计算准确率
print(f"ISOMAP Accuracy: {y_accuracy_isomap}")

y_accuracies['ISOMAP'] = y_accuracy_isomap  # 存储准确率


# 绘制准确性条形图
def draw_accuracy(models, y_accuracies, colors):
    plt.figure(figsize=(10, 5))
    plt.bar(models, list(y_accuracies.values()), color=colors, label=[f'{i}: {j:.2f}' for i, j in y_accuracies.items()])
    plt.legend(loc='best')
    plt.ylim([0, 1.01])
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Comparison of Different Dimensionality Reduction Methods with Lasso Feature Selection and GaussianNB Classifier")
    plt.show()

# 模型列表和颜色
models = ['PCA', 'KPCA', 'MDS', 'LLE', 'ISOMAP']
colors = ['r', 'c', 'b', 'm', 'g']

# 绘制图表
draw_accuracy(models, y_accuracies, colors)



