# meats_detector

## アノテーション

### 動画分割
動画をいくつかのフレームに分割して画像として保存する方法について記述します。
準備として以下のコマンドを実行して下さい。
```
pip install 'pip==18.0'
pip install pipenv
pipenv install
```

上記コマンドを実行したら、編集対象となる動画を'data/movies/'以下のディレクトリに保存して下さい。
動画を保存したら以下のコマンドを実行することで動画を分割することができます。

```
pipenv run python meats_detector/data/split_videos.py \\
--source_path data/movies/ \\
--target_path data/processed/ \\
--width None \\
--height None \\
--num_fps 0.5
```

Option説明

- source_path
動画の保存ディレクトリ、または動画ファイルを指定します。
指定しなかった場合、デフォルトで'data/movies/'以下の全ての動画が対象になります。

- target_path
分割フレームの保存ディレクトリを指定します。
指定しなかった場合、デフォルトで'data/processed/'に保存されます。
target_pathにディレクトリを指定した場合はsource_pathと同じディレクトリ構造をtarget_path内に展開し、分割フレームを保存します。

- width
保存されるフレームの幅を指定します。
デフォルトでは元動画と同じサイズで保存します。

- height
保存されるフレームの長さを指定します。
デフォルトでは元動画と同じサイズで保存します。

- num_fps
fpsあたりのフレーム取得数を指定します。
デフォルトでは0.5が指定されているため、1秒あたり0.5枚のフレーム(2秒に1枚)が取得されます。

色々とOptionをつけていますが、基本的に説明通りやる場合はOptionの指定なしでOK。
保存ディレクトリやnum_fpsのオプションを変更すると、チーム開発の際(特に実際のアノテーション時)に問題が発生する可能性あり。



### ツール

アノテーションにはLabelImgを使用します。
インストール詳細については、[LabelImg](https://github.com/tzutalin/labelImg)参照。

###  実際のアノテーション作業について
- アノテーションする動画ファイル名をslackに投げる
- 保存先は'./data/labels/<video file>/'
- 保存フォーマットはPascal.

### アノテーション時の注意点
- bounding boxを作成する際、画像外に及ばないようにする。
- 可能ならば枠ギリギリをせめない
- 複数のbounding boxを一枚の画像内で作成する場合、それぞれのbounding boxの形を似たものにする。
- 全てのbounding boxをなるべく正方形に近づけると楽かも

### ラベル詳細

- 牛
  - 生 raw_beaf
  - 半焼け half_cooked_beaf
  - 焼け cooked_beaf
  - 焦げ over_cooked_beaf

- 豚
  - 生 raw_pork
  - 半焼け half_cooked_pork
  - 焼け cooked_pork
  - 焦げ over_cooked_pork

- 鳥
  - 生 raw_chickin
  - 半焼け half_cooked_chickin
  - 焼け cooked_chickin
  - 焦げ over_cooked_chickin


現状ではSSDを用いて物体検出及び物体認識を行い、それぞれのクラスを分類する予定
HOG特徴量などの局所特徴量を用いて物体検出し、CNNを用いた分類も考えられる。
この場合は、HOGにより肉を検出すれば良いだけであるため、上記のような細かいクラス分けは必要ない。
しかし、細かいクラス分けを行うことに問題はないため、このようなラベルをつけた(後で置換すればいいだけ)


### ラベル統合
[LabelImg](https://github.com/tzutalin/labelImg)により作成されるannotation(Pascal)情報は画像毎に作成される。
しかし、このフォーマットはdlib formatとは異なるため、dlibのHOGにそのまま用いることができない。
そこでHOG特徴量を用いてbounding boxを学習させる場合は[ImageNet_Utils](https://github.com/tzutalin/ImageNet_Utils)を用いてdlib formatへ変換する。

```
./boxesCvtPascaltoDlib.py <path/to/pascal/boxes> <path/to/img/folder> --out out.xml
```
