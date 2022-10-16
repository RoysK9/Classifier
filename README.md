# 分類プログラムの内部構成

# ファイルとフォルダの説明

## utils/
役に立つツールをまとめたフォルダです

* *early_stopping.py*
    * 早期終了実装用のプログラムです。
* *icf_loss.py*
    * クラス不均衡対策Lossの1つであるICFLossを実装したプログラムです。
* *import_libraries.py*
    * よく使うライブラリ一括import用のプログラムです。
* *logger.py*
    * ターミナル上の出力をlogファイルへと保存するためのloggerを実装したプログラムです。
* *make_file_name.py*
    * ハイパーパラメータ毎のファイル名を作成するための簡単な関数が入ったプログラムです。
* *parameter_exporter.py*
    * 様々なパラメータのパターンでyamlファイルを作成するプログラムです。
* *parameter_loader.py*
    * yamlファイルからパラメータを読み込んでクラスを作成するプログラムです。
* *plot_result.py*
    * LossとAccuracyをグラフ化するプログラムです。
* *timeDistributed.py*
    * 使っていません。

## data_split.py
データセットの中のDataフォルダ内でクラス毎に分けられているデータを8:1:1に分割し, Train, Val, Testデータを作成するプログラムです。

## export_onnx.py
pthファイルをonnxファイルへ変換するプログラムです。

## learn.py
訓練のプログラムです。

## main.py
訓練，テストを実行するプログラムです。
これを実行します。

## model.sh
使用するモデルが定義されたプログラムです

## multi_balance_sample.py
バッチ内均衡化をした上でloaderを実装するプログラムです。

## multirun.sh
複数パラメータで何度も学習を実行するスクリプトです。

## mydataset.sh
データセットからloaderを作成するプログラムです。
データ拡張も行われます。
また，一部のクラスのデータを少なくすることもできます。

## seed_definer.py
シード値固定用のプログラムです。

## test.py
テストのプログラムです。
間違えた画像の出力も行います。

## 実行後作成されるフォルダ/
* *logs*
    * logファイルが置かれるフォルダです。
* *parameters*
    * yamlファイルが置かれるフォルダです。
* *weights*
    * モデルの重みファイルが置かれるフォルダです。
* *result*
    * 精度などの結果が入ったcsvファイルが置かれるフォルダです。

## データセットフォルダの中身は下記のように構成されます/
* *Dataset*
    * Data　※最低限このフォルダは用意してください。
        * 1
        * 2
        * ...
        * 9
    * train
        * 1
        * 2
        * ...
        * 9
    * val
        * trainと同様
    * test
        * trainと同様

# 使用方法
1. datasetを用意してください。
    * 参照可能な場所にデータを用意してください。mydataset.py内とmulti_balance_sample.py内のpath変数も変更してください。
    * multi_balance_sample.pyを使用する場合はそちらのpath変数も変更してください。
2. Train，Val，Testに未分類でかつDataフォルダ内はクラス毎に分類できている場合，data_split.pyを実行することでデータセットを作成することができます。
    * python data_split.py で実行可能です。
3. utilsフォルダに入ってparameter_exporter.pyを実行してください。その後utilsフォルダから出てください。
    * python parameter_exporter.py で実行可能です。
4. 全パラメータで学習したい場合は，multirun.shを実行してください。 
    * linuxであればbash multirun.sh 1 0 で実行できます。
    * 1は使用可能なPCの数，0はプログラムを動かすPCが何番目のPCかを意味しています。
5. 特定のparameterで学習したい場合は，該当するyamlファイル名を付与してmain.pyを実行してください。
    * 例. python main.py --yaml_name param0 
6. onnxファイルを出力したい場合，export_onnx.pyを実行してください。その際，どのパラメータで学習したモデルを使うかを指定する必要があります。
    * 例. python export_onnx.py --yaml_name param0 
