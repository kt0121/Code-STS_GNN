import json

import fasttext
from datasets import load_dataset
from tqdm import tqdm

from utils import MyEncoder, get_graph_attr

dataset = load_dataset("sick")

model = fasttext.load_model("./create_graphs/w2v-model/crawl-300d-2M-subword.bin")

stop_words = ["a", "and", "of", "to", ",", "s", "t", ""]

for i, data in enumerate(tqdm(dataset["test"])):
    nodes_1, edges_1 = get_graph_attr(data["sentence_A"], model, stop_words)
    nodes_2, edges_2 = get_graph_attr(data["sentence_B"], model, stop_words)
    with open(f"./dataset/test/{i}.json", mode="wt", encoding="utf-8") as file:
        json.dump(
            {
                "edges_1": edges_1,
                "edges_2": edges_2,
                "features_1": nodes_1,
                "features_2": nodes_2,
                "relation_score": data["relatedness_score"],
            },
            file,
            ensure_ascii=False,
            cls=MyEncoder,
        )

for i, data in enumerate(tqdm(dataset["train"])):
    nodes_1, edges_1 = get_graph_attr(data["sentence_A"], model, stop_words)
    nodes_2, edges_2 = get_graph_attr(data["sentence_B"], model, stop_words)
    with open(f"./dataset/train/{i}.json", mode="wt", encoding="utf-8") as file:
        json.dump(
            {
                "edges_1": edges_1,
                "edges_2": edges_2,
                "features_1": nodes_1,
                "features_2": nodes_2,
                "relation_score": data["relatedness_score"],
            },
            file,
            ensure_ascii=False,
            cls=MyEncoder,
        )
