import json
import os
import shutil
from typing import List, Tuple

import numpy
import stanza
from tqdm import tqdm

try:
    nlp = stanza.Pipeline("en", processors="tokenize,mwt,pos,lemma,depparse")
except Exception:
    stanza.download("en")
    nlp = stanza.Pipeline("en", processors="tokenize,mwt,pos,lemma,depparse")


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)


def get_graph_attr(sentence, model, stop_words):
    lemmas = []
    edges_raw = []
    lemmas_index = {}
    count_stop_words = 0
    doc = nlp(sentence)
    sent = doc.sentences[0]
    for word in sent.words:
        if word.lemma not in stop_words:
            lemmas.append(model[word.lemma])
            lemmas_index[word.lemma] = word.id - 1 - count_stop_words
            word.head != 0 and sent.words[
                word.head - 1
            ].lemma not in stop_words and edges_raw.append(
                [word.lemma, sent.words[word.head - 1].lemma]
            )
        else:
            count_stop_words += 1

    edges = list(map(lambda x: [lemmas_index[x[0]], lemmas_index[x[1]]], edges_raw))
    return (lemmas, edges)


def create_graphs(
    dataset,
    model,
    data_keys: Tuple[str, str, str],
    stop_words: List[str],
    save_dir: str,
):
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    os.makedirs(save_dir, exist_ok=True)

    for i, data in enumerate(tqdm(dataset)):
        nodes_1, edges_1 = get_graph_attr(data[data_keys[0]], model, stop_words)
        nodes_2, edges_2 = get_graph_attr(data[data_keys[1]], model, stop_words)
        with open(f"{save_dir}/{i}.json", mode="wt", encoding="utf-8") as file:
            json.dump(
                {
                    "edges_1": edges_1,
                    "edges_2": edges_2,
                    "features_1": nodes_1,
                    "features_2": nodes_2,
                    "relation_score": data[data_keys[2]],
                },
                file,
                ensure_ascii=False,
                cls=NumpyEncoder,
            )
