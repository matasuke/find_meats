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
--num_fps 1
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
デフォルトでは1が指定されているため、1秒あたり1枚のフレームが取得されます。

色々とOptionをつけていますが、基本的に説明通りやる場合はOptionの指定なしでOK。


### ツール

アノテーションにはLabelImgを使用します。
インストール詳細については、[LabelImg](https://github.com/tzutalin/labelImg)参照。


