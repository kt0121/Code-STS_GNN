import json

import numpy
import stanza

try:
    nlp = stanza.Pipeline("en", processors="tokenize,mwt,pos,lemma,depparse")
except Exception:
    stanza.download("en")
    nlp = stanza.Pipeline("en", processors="tokenize,mwt,pos,lemma,depparse")


def get_graph_attr(sentence, model, stop_words_list):
    lemmas = []
    edges_raw = []
    lemmas_index = {}
    count_stop_words = 0
    doc = nlp(sentence)
    sent = doc.sentences[0]
    for word in sent.words:
        if word.lemma not in stop_words_list:
            lemmas.append(model[word.lemma])
            lemmas_index[word.lemma] = word.id - 1 - count_stop_words
            word.head != 0 and edges_raw.append(
                [word.lemma, sent.words[word.head - 1].lemma]
            )
        else:
            count_stop_words += 1

    edges = list(map(lambda x: [lemmas_index[x[0]], lemmas_index[x[1]]], edges_raw))
    return (lemmas, edges)


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)
