import numpy as np
import requests
import json

from gensim.models import KeyedVectors
model = KeyedVectors.load_word2vec_format('./app/resources/word2vec-model.bin', binary=True)


class InfoExtractor:

    # def __init__(self):
    #     self.properties = {'openie.affinity_probability_cap': 2 / 3, }

    def extract_tuples(self, text: str) -> str:
        tuples = []
        url = 'http://localhost:9000/'
        params = {'properties': '{"annotators": "tokenize,ssplit,pos,lemma,depparse,natlog,openie", "splitter.disable" : "true",   "openie.max_entailments_per_clause": "1"}',
                  'openie.affinity_probability_cap': 2 / 3, "openie.triple.strict": "true"}
        # a rabbit is under a house. a panda is far away from the house. a tree is next to the panda
        # Get information about the sentence from CoreNLP
        r = requests.post(url, data=text, params=params, timeout=60)
        print(r.text)
        data = json.loads(r.text)
        for sentence in data["sentences"]:
            print(sentence.keys())
            for triple in sentence["openie"]:
                print(triple['subjectSpan'])
                print(triple['objectSpan'])
                print(triple['relationSpan'])
                tokens = dict()
                tokens['sub'] = triple['subject']
                tokens["obj"] = triple['object']
                tokens['relation'] = triple['relation']

                print(tokens)
                tuples.append(tokens)
        return json.dumps({"array": tuples})

    def sentenceToGraphData(self, tuples: str):

        print(tuples)
        # tuples = tuples.replace("%22", '"')

        data = json.loads(tuples)
        node_words = set()
        for relation in data['array']:
            node_words.add(relation['sub'])
            node_words.add(relation['obj'])

        word_embedding_size = 300
        node_words = list(node_words)
        node_features = np.empty((len(node_words), word_embedding_size), dtype=float)
        edge_words = []
        edge_features = np.empty((len(data['array']), word_embedding_size), dtype=float)
        edges_u = []
        edges_v = []
        for relation in data['array']:
            sub_index = node_words.index(relation['sub'])
            obj_index = node_words.index(relation['obj'])

            node_features[sub_index] = model[node_words[sub_index]].tolist()
            node_features[obj_index] = model[node_words[obj_index]].tolist()

            edges_u.append(sub_index)
            edges_v.append(obj_index)

            edge_words.append(relation['relation'])

            for words in edge_words:
                embedding = None
                for word in words.split():
                    if embedding is None and model.__contains__(word):
                        embedding = np.array(model[word])
                    elif model.__contains__(word):
                        embedding += np.array(model[word])
                edge_features[len(edge_words) - 1] = embedding.tolist()

        return node_words, node_features, edge_words, edge_features, edges_u, edges_v
