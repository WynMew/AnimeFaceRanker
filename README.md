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
