# 卒論のコード

卒論で使ったコードです

## 環境

Cuda: 11.2
cudnn: 8.1
python3.8

## 準備

- https://fasttext.cc/docs/en/english-vectors.html から crawl-300d-2M-subword.zip をダウンロードし、create_graphs/src/w2v-model/に置く
- conda-env.yaml を元に conda 仮想環境を作成

## 実行

### グラフの作成

create_graphs 以下

```(bash)
python create_graphs/src/main.py
```

### 学習

calc_similarity 以下

ここは　https://github.com/benedekrozemberczki/SimGNN　を元に作られています

```(bash)
python calc_similarity/src/main.py (-hをつけるとオプションが表示されれる)
```

論文中の全パターンを試すには以下

```(bash)
nohup sh train_all.sh > <LOG_DIR> &
```
