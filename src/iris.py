from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report
import joblib


# data の取得
iris = load_iris()

# 入力、出力変数に切り分け
x = iris.data
t = iris.target

# 学習とテストデータに分割
x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=0.3, random_state=0)

# インスタンス化
model = svm.LinearSVC()

# 学習
model.fit(x_train, t_train)
pred = model.predict(x_test)

# 予測の確認
print(classification_report(t_test, pred))

# 学習済みモデルの保存
joblib.dump(model, "src/iris.pkl", compress=True)