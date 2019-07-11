[![996.icu](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu)

# AnimeFaceRanker
Check detected Anime Face Quality (delete noise data). Implementated with [HRNet](https://github.com/HRNet/HRNet-Image-Classification)

## Dependencies
- Python 3.6+ (Anaconda)
- PyTorch-0.4.0 +
- scipy, numpy, sklearn etc.
- OpenCV3 (Python)

## Usage
- Download Danbooru dataset(or any other dataset you want to process)

- face detection with [lbpcascade_animeface](https://github.com/nagadomi/lbpcascade_animeface) or [danbooru-utility](https://github.com/reidsanders/danbooru-utility)

- Make your training dataset by hand (Good face detection and bad case, a few thousand each). You may skip this.

- Training with FaceClassifier.py. You may skip this.

- Rank your face detection with FaceRanker.py

- My [trained model](https://pan.baidu.com/s/1s71jxxTahzT0eyAWtlodhQ). 提取码: hjpn; Eatraction code: hjpn

Good Face Samples: ![alt text](https://pic4.zhimg.com/80/v2-61c43d15ff81f9930597e9c6ce8bf65b_hd.jpg)

Bad Case: ![alt text](https://pic4.zhimg.com/80/v2-55ba608d71e45baa4a91f51943d23827_hd.jpg)
