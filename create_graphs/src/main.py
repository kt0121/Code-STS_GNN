import fasttext
from datasets import load_dataset

from utils import create_graphs

# sick_train = load_dataset("sick", split="train")
# sick_test = load_dataset("sick", split="test")
# sick_validation = load_dataset("sick", split="validation")
sts_b_train = load_dataset("stsb_multi_mt", name="en", split="test")
sts_b_test = load_dataset("stsb_multi_mt", name="en", split="dev")
sts_b_validation = load_dataset("stsb_multi_mt", name="en", split="train")

model = fasttext.load_model("./create_graphs/w2v-model/crawl-300d-2M-subword.bin")

stop_words = ["a", "an", "the", ",", ""]

# create_graphs(
#     dataset=sick_train,
#     model=model,
#     data_keys=("sentence_A", "sentence_B", "relatedness_score"),
#     stop_words=stop_words,
#     save_dir="./dataset/SICK/train",
# )
# create_graphs(
#     dataset=sick_test,
#     model=model,
#     data_keys=("sentence_A", "sentence_B", "relatedness_score"),
#     stop_words=stop_words,
#     save_dir="./dataset/SICK/test",
# )
# create_graphs(
#     dataset=sick_validation,
#     model=model,
#     data_keys=("sentence_A", "sentence_B", "relatedness_score"),
#     stop_words=stop_words,
#     save_dir="./dataset/SICK/validation",
# )


create_graphs(
    dataset=sts_b_train,
    model=model,
    data_keys=("sentence1", "sentence2", "similarity_score"),
    stop_words=[],
    save_dir="./dataset/STS-B/train",
)
create_graphs(
    dataset=sts_b_test,
    model=model,
    data_keys=("sentence1", "sentence2", "similarity_score"),
    stop_words=[],
    save_dir="./dataset/STS-B/test",
)
create_graphs(
    dataset=sts_b_validation,
    model=model,
    data_keys=("sentence1", "sentence2", "similarity_score"),
    stop_words=[],
    save_dir="./dataset/STS-B/validation",
)
