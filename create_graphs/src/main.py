import fasttext
from datasets import load_dataset

from utils import create_dataset

sick_train = load_dataset("sick", split="train")
sick_test = load_dataset("sick", split="test")
sick_validation = load_dataset("sick", split="validation")
sick_keys = ("sentence_A", "sentence_B", "relatedness_score")

sts_b_train = load_dataset("stsb_multi_mt", name="en", split="test")
sts_b_test = load_dataset("stsb_multi_mt", name="en", split="dev")
sts_b_validation = load_dataset("stsb_multi_mt", name="en", split="train")
sts_keys = ("sentence1", "sentence2", "similarity_score")

model = fasttext.load_model("./create_graphs/w2v-model/crawl-300d-2M-subword.bin")

stop_words = ["a", "an", "the", "", "."]
create_dataset(
    train_dataset=sick_train,
    test_dataset=sick_test,
    validation_dataset=sick_validation,
    model=model,
    data_keys=sick_keys,
    stop_words=stop_words,
    save_dir="./dataset/w2v/SICK",
    # use_bert=True,
    is_adjacent=False,
)

create_dataset(
    train_dataset=sick_train,
    test_dataset=sick_test,
    validation_dataset=sick_validation,
    model=model,
    data_keys=sick_keys,
    stop_words=stop_words,
    save_dir="./dataset/w2v/SICK-adjacent",
    # use_bert=True,
    is_adjacent=True,
)


create_dataset(
    train_dataset=sts_b_train,
    test_dataset=sts_b_test,
    validation_dataset=sts_b_validation,
    model=model,
    data_keys=sts_keys,
    stop_words=stop_words,
    save_dir="./dataset/w2v/STSb",
    # use_bert=True,
    is_adjacent=False,
)

create_dataset(
    train_dataset=sts_b_train,
    test_dataset=sts_b_test,
    validation_dataset=sts_b_validation,
    model=model,
    data_keys=sts_keys,
    stop_words=stop_words,
    save_dir="./dataset/w2v/STSb-adjacent",
    # use_bert=True,
    is_adjacent=True,
)
