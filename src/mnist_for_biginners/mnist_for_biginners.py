# -*- coding: utf-8 -*-

# TensowFlowのインポート
import tensorflow as tf
# MNISTを読み込むためinput_data.pyを同じディレクトリに置きインポートする
# input_data.pyはチュートリアル内にリンクがあるのでそこから取得する
# https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/examples/tutorials/mnist/input_data.py
import input_data

import time

# parameters
learning_rate = 0.01
training_epochs = 1000

# 開始時刻
start_time = time.time()
print ("開始時刻: " + str(start_time))

# MNISTデータの読み込み
# 60000点の訓練データ（mnist.train）と10000点のテストデータ（mnist.test）がある
# 訓練データとテストデータにはそれぞれ0-9の画像とそれに対応するラベル（0-9）がある
# 画像は28x28px(=784)のサイズ
# mnist.train.imagesは[60000, 784]の配列であり、mnist.train.lablesは[60000, 10]の配列
# lablesの配列は、対応するimagesの画像が3の数字であるならば、[0,0,0,1,0,0,0,0,0,0]となっている
# mnist.test.imagesは[10000, 784]の配列であり、mnist.test.lablesは[10000, 10]の配列
print("--- MNISTデータの読み込み開始 ---")
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print("mnist.train: ", end="")
print(mnist.train.images.shape)
print(mnist.train.images)
print("mnist.test: ", end="")
print(mnist.train.images.shape)
print(mnist.test.images)
print("--- MNISTデータの読み込み完了 ---")

# 訓練画像を入れる変数
# 訓練画像は28x28pxであり、これらを1行784列のベクトルに並び替え格納する
# Noneとなっているのは訓練画像がいくつでも入れられるようにするため
x = tf.placeholder(tf.float32, [None, 784], name="x")

# yは正解データのラベル
y = tf.placeholder(tf.float32, [None, 10], name="y")

# 重み
# 訓練画像のpx数の行、ラベル（0-9の数字の個数）数の列の行列
# 初期値として0を入れておく
W = tf.Variable(tf.zeros([784, 10]), name="weights")

# バイアス
# ラベル数の列の行列
# 初期値として0を入れておく
b = tf.Variable(tf.zeros([10]), name="bias")

# ソフトマックス回帰を実行
# yは入力x（画像）に対しそれがある数字である確率の分布
# matmul関数で行列xとWの掛け算を行った後、bを加算する。
# yは[1, 10]の行列
activation = tf.nn.softmax(tf.matmul(x, W) + b)

#   クロスエントロピーの計算をname_scopeでまとめる
with tf.name_scope("cross-entropy") as scope:
    cross_entropy = -tf.reduce_sum(y*tf.log(activation))

#   最急降下法の計算をname_scopeでまとめる
with tf.name_scope("training") as scope:
    # 勾配硬化法を用い交差エントロピーが最小となるようyを最適化する
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# TensorBoardにまとめる項目の設定
tf.scalar_summary("lossの変化", cross_entropy)

# 用意した変数Veriableの初期化を実行する
init = tf.initialize_all_variables()

# Sessionを開始する
# runすることで初めて実行開始される（run(init)しないとinitが実行されない）

sess = tf.Session()
sess.run(init)

# TensorBoardにこのネットワークのGraphを描画できるように
# TensorBoardで表示する値の設定
summary_op = tf.merge_all_summaries()
summary_writer = tf.train.SummaryWriter('tb_mnist_biginners', graph=sess.graph)

# 正しいかの予測
# 計算された画像がどの数字であるかの予測yと正解ラベルy_を比較する
# 同じ値であればTrueが返される
# argmaxは配列の中で一番値の大きい箇所のindexが返される
# 一番値が大きいindexということは、それがその数字である確率が一番大きいということ
# Trueが返ってくるということは訓練した結果と回答が同じということ
correct_prediction = tf.equal(tf.argmax(activation,1), tf.argmax(y,1))

# 精度の計算
# correct_predictionはbooleanなのでfloatにキャストし、平均値を計算する
# Trueならば1、Falseならば0に変換される
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


# 1000回の訓練（train_step）を実行する
# next_batch(100)で100つのランダムな訓練セット（画像と対応するラベル）を選択する
# 訓練データは60000点あるので全て使いたいところだが費用つまり時間がかかるのでランダムな100つを使う
# 100つでも同じような結果を得ることができる
# feed_dictでplaceholderに値を入力することができる
print("--- 訓練開始 ---")
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

    if i%100 == 0:
        print("step %d, training accuracy %g"%(i, sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys})))

    # 1step毎にTensorBoardに表示する値を追加する
    summary_str = sess.run(summary_op, feed_dict={x :batch_xs, y: batch_ys})
    summary_writer.add_summary(summary_str, i)
print("--- 訓練終了 ---")

#
# 精度の実行と表示
# テストデータの画像とラベルで精度を確認する
# ソフトマックス回帰によってWとbの値が計算されているので、xを入力することでyが計算できる
print("精度")
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))

# 終了時刻
end_time = time.time()
print("終了時刻: " + str(end_time))
print("かかった時間: " + str(end_time - start_time))
