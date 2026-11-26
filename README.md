# Pコン跡の判定

ResNet-18の3モデル等重み平均によるPコン跡判定の再現手順です。

## 動作環境

Python 3.10または3.11を使用します。Apple SiliconのMacでは利用可能な場合にMPSを使い、それ以外のMacではCPUで実行します。全15モデルを学習するため、所要時間は機器によって異なります。 [PyTorchのMPSの説明](https://docs.pytorch.org/docs/stable/notes/mps.html)

Apple SiliconのMacでは、リポジトリ内で次を実行します。

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.11.0 torchvision==0.26.0
python -m pip install -r requirements.txt
```

Intel Macでは、上のPyTorchのインストール行を `python -m pip install torch==2.2.2 torchvision==0.17.2` に置き換えます。 [PyTorchの配布バージョン](https://pytorch.org/get-started/previous-versions/)

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
