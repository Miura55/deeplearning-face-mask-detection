# deeplearning-face-mask-detection

## Install

```
$ pip3 install -r requirements.txt
```

## How to Use

### データセットの用意

レポジトリ上に`data`ディレクトリを用意し、マスクをつけた顔の画像、マスクをつけていない顔の画像のデータセットを用意する

```
./
├─data
│  ├─without_mask
│  └─with_mask
```

[cabani/MaskedFace-Net](https://github.com/cabani/MaskedFace-Net) - マスクをつけた顔画像のデータセットを取得

[Real and Fake Face Detection](https://www.kaggle.com/ciplab/real-and-fake-face-detection) - マスクをつけていない顔画像のデータセットを取得(Real のデータを使用)

### 学習

`jupyter-notebook`を起動し、`train_mask_detect.ipynb`または`train_mobilenetV2.ipynb`を使って学習を行う。

`train_mask_detect.ipynb`は[Classifire10 のチュートリアル](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)のモデルを参考に作成したもの。

`train_mobilenetV2.ipynb`は[MobileNetV2](https://pytorch.org/hub/pytorch_vision_mobilenet_v2/)を転移学習させたもの。

### モデルの保存先

`examples/face_mask_detector.pth` - pth 形式のモデル(Pytorch で読み込む用)

`examples/face_mask_detector.onnx` - onnx 形式のモデル(OpenCV で推論を行う用)

### モデルの実行(画像)

```
python detect_mask.py
```

### モデルの実行(Web カメラ)

```
python detect_mask_webcam.py
```
