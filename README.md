# Pコン跡の判定

ResNet-18の3モデル等重み平均によるPコン跡判定の再現手順です。

## 動作環境

macOS 13以降とPython 3.10または3.11を使用します。Apple SiliconとIntel Macで手順は共通です。リポジトリ内で次を実行します。

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Apple Siliconでは利用可能な場合にMPSで学習し、それ以外のMacではCPUで学習します。

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

## 学習と提出CSVの生成

```sh
python reproduce.py
```

モデルを学習し、提出CSVを `outputs/submission_resnet18_raw_median_sharpen_aug.csv` に保存します。

## 学習済みの重みで推論する

リポジトリ内で提出時の重みを展開し、壁面画像をCPUで判定します。

```sh
curl -fL https://github.com/MotoiHAGIHARA/AI-FDE-ResNet18-Submission/releases/download/submission-weights-v1/resnet18-weights.tar.gz | tar -xz
python reproduce.py --device cpu --output submission-weights classify wall.jpg
```

スコアと判定（Pコン跡ありは `1`、なしは `0`）を表示します。256×256以外の画像は、縦横比を保って余白を足し、256×256にそろえます。
