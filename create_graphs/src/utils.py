import json
import os
import shutil
from typing import List, Tuple

import numpy
import stanza

# import torch
from tqdm import tqdm

# from transformers import BertModel, BertTokenizer

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# torch.set_default_tensor_type("torch.cuda.FloatTensor")
try:
    nlp = stanza.Pipeline(
        "en", processors="tokenize,mwt,pos,lemma,depparse", tokenize_no_ssplit=True
    )
except Exception:
    stanza.download("en")
    nlp = stanza.Pipeline(
        "en", processors="tokenize,mwt,pos,lemma,depparse", tokenize_no_ssplit=True
    )

# tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
# bert_model = BertModel.from_pretrained("bert-large-uncased").to(device)


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


def get_graph_attr(sentence, model, stop_words, use_bert, is_adjacent=False):
    lemmas = []
    edges_raw = []
    lemmas_index = {}
    count_stop_words = 0
    doc = nlp(sentence)
    sent = doc.sentences[0]
    # if use_bert:
    #     encoded_input = tokenizer.encode(sentence, return_tensors="pt")
    #     input_ids = torch.tensor(encoded_input["input_ids"], device=device)
    #     output = bert_model(input_ids).last_hidden_state[0]
    for i, word in enumerate(sent.words):
        if word.lemma not in stop_words:
            # if use_bert:
            #     lemmas.append(output[i + 1])
            # else:
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
    is_adjacent and edges.extend([[i, i + 1] for i in range(len(lemmas) - 1)])
    return (lemmas, edges)


def get_similarity_snli(label):
    if label == 2:
        return 0.0
    elif label == 0:
        return 5.0
    elif label == 1:
        return 3.0
    else:
        return 1.0


def create_graphs(
    dataset,
    model,
    data_keys: Tuple[str, str, str],
    stop_words: List[str],
    save_dir: str,
    use_bert: bool,
    is_adjacent: bool,
):
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    os.makedirs(save_dir, exist_ok=True)

    for i, data in enumerate(tqdm(dataset)):
        if data_keys[2] != "label" and data[data_keys[2]] == -1:
            continue
        nodes_1, edges_1 = get_graph_attr(
            sentence=data[data_keys[0]],
            model=model,
            stop_words=stop_words,
            use_bert=use_bert,
            is_adjacent=is_adjacent,
        )
        nodes_2, edges_2 = get_graph_attr(
            sentence=data[data_keys[1]],
            model=model,
            stop_words=stop_words,
            use_bert=use_bert,
            is_adjacent=is_adjacent,
        )
        with open(f"{save_dir}/{i}.json", mode="wt", encoding="utf-8") as file:
            json.dump(
                {
                    "edges_1": edges_1,
                    "edges_2": edges_2,
                    "features_1": nodes_1,
                    "features_2": nodes_2,
                    "relation_score": data[data_keys[2]]
                    if data_keys[2] != "label"
                    else get_similarity_snli(data[data_keys[2]]),
                },
                file,
                ensure_ascii=False,
                cls=NumpyEncoder,
            )


def create_dataset(
    train_dataset,
    test_dataset,
    validation_dataset,
    model,
    data_keys,
    stop_words,
    save_dir,
    use_bert=False,
    is_adjacent=False,
):
    create_graphs(
        dataset=train_dataset,
        model=model,
        data_keys=data_keys,
        stop_words=stop_words,
        save_dir=f"{save_dir}/train/",
        use_bert=use_bert,
        is_adjacent=is_adjacent,
    )
    create_graphs(
        dataset=test_dataset,
        model=model,
        data_keys=data_keys,
        stop_words=stop_words,
        save_dir=f"{save_dir}/test/",
        use_bert=use_bert,
        is_adjacent=is_adjacent,
    )
    create_graphs(
        dataset=validation_dataset,
        model=model,
        data_keys=data_keys,
        stop_words=stop_words,
        save_dir=f"{save_dir}/validation/",
        use_bert=use_bert,
        is_adjacent=is_adjacent,
    )
