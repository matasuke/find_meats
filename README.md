# find_meats

## タスク分割
1. アノテーション
- 動画及び画像に対するアノテーション作業
- どの程度データを必要か？

2. 肉検出器の作成
  - 肉のクラスも取得する
  - Movidiusもあるので、deepベースの手法でfine-tuningしてみる
  - HOGなど非E2Eでもやってみても良い

3. やけ具合分類器
  - 2の結果から焼け具合を識別するnetwork構築
  - 物体検出(SSD)による2と3の統合も可能

4. 焼き時間測定
  - 2の結果から肉を認識後、焼き時間を計測する
  - 肉腫から反転タイミングをリコメンドできる。
  - カメラが動いてもどの肉がどの肉に対応するか記録する仕組みが必要？

5. 反転検出器
  - 肉の反転を検出する

6. 裏面焼け具合推定
  - 2の肉腫、4の焼き時間，5の反転タイミングから焼け具合を測定する検出機を作成する

7. アプリ開発
  - それぞれのモジュールを統合し、androidアプリとして開発

## データ収集
データセット構築、学習を考慮すると、画像のアスペクト比は統一する必要がある。
アスペクト比16:9で撮影。
写真撮影時はブレ、はみ出しに気をつける。
肉の一部がはみ出していると後述するアノテーション作業時に面倒が生じるため、一切はみ出さないように。

収集データ

各データ網の上、皿の上などのパターンがほしい。

- 牛肉(とりえあずはカルビ限定)
  - 生
  - 半生
  - 焼け
  - 焦げ
- 豚肉(豚バラ限定)
  - 生
  - 半生
  - 焼け
  - 焦げ
- 鶏肉(鶏もも限定)
  - 生
  - 半生
  - 焼け
  - 焦げ


## アノテーション

### 動画分割
動画をいくつかのフレームに分割して画像として保存する方法について記述します。
準備として以下のコマンドを実行して下さい。
```
brew install python3
pip install 'pip==18.0'
pip install pipenv
pipenv install
```

上記コマンドを実行したら、編集対象となる動画を'data/movies/'以下のディレクトリに保存して下さい。
動画の保存先となるサブディレクトリ構造はgoogle driveにアップロードされている動画のディレクトリ構造と同じものにして下さい。
google driveのCICPディレクトリ以下に09_12/ 06_19などのディレクトリがあるとしたら、./data/movies/以下のディレクトリに09_12/ 06_19/を作成する要領
動画を保存したら以下のコマンドを実行することで動画を分割することができます。

```
pipenv run python find_meats/data/split_videos.py \\
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

#### Hotkeys

|   keys   |                    detail                 |
|----------|-------------------------------------------|
| Ctrl + u |  Load all of the images from a directory  |
| Ctrl + r | Change the default annotation target dir  |
| Ctrl + s |                  Save                     |
| Ctrl + d |  Copy the current label and rect box      |
|  Space   |  Flag the current image as verified       |
|     w    |             Create a rect box             |
|     d    |                 Next image                |
|     a    |              Previous image               |
|   del    |       Delete the selected rect box        |
|  Ctrl++  |                 Zoom in                   |
|  Ctrl--  |                 Zoom out                  |
|  ↑→↓←    | Keyboard arrows to move selected rect box |


###  実際のアノテーション作業について
- アノテーションする画像名をslackに投げる
- 保存先は'./data/labels/video_name/'
- 保存フォーマットはPascal.

### アノテーション時の注意点
- [VOC2007 annotation guidelines](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/guidelines.html)のannotation guidelinesに基づきアノテーションを行う。

- bounding boxは対象の全ピクセルを含む必要があるが、極度に大きすぎないように
- bounding boxは画像からはみ出さないように

#### 報告事項
以下に該当する画像を発見した場合はslackに画像名と共に報告
- occlusion(何かと重なって半分だけ見えている)
- 肉の画像が画像からはみ出ているもの
- 画像がボケているもの
- 肉の種類判別が難しいもの

#### ラベル対象の肉
- 後述するラベル詳細に記載される全ての肉
- めちゃくちゃ小さい肉以外
- 10~20%以上が見えている肉


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
  - 生 raw_chicken
  - 半焼け half_cooked_chicken
  - 焼け cooked_chicken
  - 焦げ over_cooked_chicken


'labelImg/data/predefined_classes.txt'を書き換えることで、Labelを特定のものだけに指定することが可能。
'data/labelimg/predefined_classes.txt'を上記のものと置き換えると良い。

```
cp data/labelimg/predefined_classes.txt  <path>/<to>/<labelimg/data/predefined_classes.txt>
```


現状ではSSDを用いて物体検出及び物体認識を行い、それぞれのクラスを分類する予定
HOG特徴量などの局所特徴量を用いて物体検出し、CNNを用いた分類も考えられる。
この場合は、HOGにより肉を検出すれば良いだけであるため、上記のような細かいクラス分けは必要ない。
しかし細かいクラス分けを行うことに問題はないため、このようなラベルをつけた(後で置換すればいいだけ)


### ラベル統合
[LabelImg](https://github.com/tzutalin/labelImg)により作成されるannotation(Pascal)情報は画像毎に作成される。
しかし、このフォーマットはdlib formatとは異なるため、dlibのHOGにそのまま用いることができない。
そこでHOG特徴量を用いてbounding boxを学習させる場合は[ImageNet_Utils](https://github.com/tzutalin/ImageNet_Utils)を用いてdlib formatへ変換する。

```
./boxesCvtPascaltoDlib.py <path/to/pascal/boxes> <path/to/img/folder> --out out.xml
```

## pull requests
pull requestsを送る際は、事前にtestを実行して下さい。
本レポジトリのコードはflake8, mypy, pytestを採用しています。
コードを追加した際は'./test/lint'を実行してエラーが発生しないことを確認して下さい。


testコードを書いた場合は、'./test/run'を実装して全てのテストケースを突破できることを確認して下さい。


pull requestsのコードレビューを受けてOKが出た場合は、自身でmergeしてください。
