# Pコン跡の判定

コンクリート壁面画像にPコン跡が含まれるかを判定するコンペの提出コードです。生画像、メディアンフィルタ画像、先鋭化画像で個別に学習したResNet-18の予測を等重みで平均し、提出CSVを生成します。画像、配布ラベル、学習済み重みは各自の環境で用意・生成します。

## 動作環境

Python 3.10または3.11を使用します。Apple SiliconのMacでは利用可能な場合にMPSを使い、それ以外のMacではCPUで実行します。全15モデルを学習するため、所要時間は機器によって異なります。 [PyTorchのMPSの説明](https://docs.pytorch.org/docs/stable/notes/mps.html)

Apple SiliconのMacでは、リポジトリ内で次を実行します。

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.11.0 torchvision==0.26.0
python -m pip install -r requirements.txt
```

Intel Macでは、上のPyTorchのインストール行を `python -m pip install torch==2.2.2 torchvision==0.17.2` に置き換えます。CUDA 12.8版のPyTorchを使う環境では、Apple Silicon向けのPyTorchのインストール行に `--index-url https://download.pytorch.org/whl/cu128` を付けます。 [PyTorchの配布バージョン](https://pytorch.org/get-started/previous-versions/)

## データの配置

コンペ配布の `train_label.csv`、`test.txt`、`sample_submission.csv` を `data/` に置きます。SDNET2018から壁面画像の `W/` を展開します。ファイル名とディレクトリの大文字・小文字は配布時のままにします。 [SDNET2018の公式配布ページ](https://digitalcommons.usu.edu/all_datasets/48/)

```text
data/
├── train_label.csv
├── test.txt
├── sample_submission.csv
└── SDNET2018/
    └── W/
        ├── CW/
        └── UW/
```

## 解答手順

```sh
python reproduce.py
```

配布された学習ラベルから元写真単位の5分割を再生成し、生画像・メディアン・先鋭化の各条件で5モデルずつ、計15モデルをランダム初期化から学習します。各条件のテスト予測は5モデルで平均し、さらに3条件の予測を等重みで平均します。判定閾値は、各画像を学習に含まない分割で得た検証予測から、F1が最大になるように決めます。

出力は `outputs/submission_resnet18_raw_median_sharpen_aug.csv` です。CSVは `test.txt` の順序で、ヘッダーなしの `画像パス,0または1` 形式です。Pコン跡ありを `1` とします。検証結果と生成したCSVのSHA-256は `outputs/metrics.json` に記録します。

CPUを明示するには `python reproduce.py --device cpu` を実行します。画像の配置が異なる場合は、`--images` に `W/` を含むディレクトリを指定します。メモリに合わせてバッチサイズを下げる場合は、別の出力先を指定し、たとえば `python reproduce.py --batch-size 16 --output outputs-small-batch` を実行します。中断後に同じ設定で起動すると、完了済みの学習を再利用します。学習が完了している場合は `python reproduce.py predict` で推論だけを再実行できます。

元の実行では、検証F1が **0.86146** で、提出CSVは8,059行、陽性予測は156件でした。CSVのSHA-256は `150430e27aec276ac49e9ee20582e059ca21fe292f62f86a97e0eb2a32e0521e` です。デバイスやライブラリ、バッチサイズの違いにより、再学習時のスコアやCSVが変わる場合があります。

## 創意工夫

生画像モデルの失敗例から、薄い縁や画像の端で切れたPコン跡を見逃し、小さな穴や欠けを誤検出することが分かりました。ハフ変換と円形度・輪郭の滑らかさを使うルールも検証しました。薄い跡や一部分しか見えない跡では輪郭の抽出が不安定で、欠けや小さな穴にも楕円が当てはまる例がありました。元写真単位の5分割検証で、試したルールベースの最高F1は0.677でした。

そこで、輪郭に加えて穴の内側の陰影や壁面の質感も判定に使えるよう、生画像、3×3メディアンフィルタ画像、アンシャープマスクによる先鋭化画像をそれぞれ使って学習しました。先鋭化モデルでは、学習時にアンシャープマスクの強さを0.2〜0.8で変え、推論時は0.5に固定しました。3条件の予測を平均することで、輪郭が見えやすい画像と、周辺の濃淡や質感が手がかりになる画像の両方を扱いました。
