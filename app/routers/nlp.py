DIRECTION_CLASSES = 6
DISTANCE_CLASSES = 3

import copy
from distutils.log import info
from msilib.schema import Class
from typing import Optional
from fastapi import APIRouter
from pydantic import BaseModel
import numpy as np
import nltk
import json
import speech_recognition as sr
from pydantic import BaseModel

from app.nlp.information_extraction import InfoExtractor
# from app.main import distance_model
# from app.main import direction_model

# with open("./app/resources/dataset.json", "w") as dataset_write:
#     global json_store_file
#     json_store_file = dataset_write


base_animate_words_file = open("./app/resources/base_animate_words", "r")
base_animate_words = [word for line in base_animate_words_file for word in line.split()]

class NLPEngine:

    def __init__(self):
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')


    def generate_pos_tags(self, text: str):
        to_return = TaggedSpeech()
        # tokenize the sentence
        tokens = nltk.word_tokenize(text)
        print("Tokens", tokens)
        # do POS tagging on it
        tags = nltk.pos_tag(tokens=tokens)
        print("POS tags", tags)
        to_return.tokens = tokens
        to_return.tags = tags
        return to_return

    def get_speech_and_convert_to_text(self):
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("Talk")
            audio_text = r.listen(source)
            print("Time over, thanks")
            # recognize_() method will throw a request error if the API is unreachable, hence using exception handling
            try:
                # using google speech recognition
                print("Text: " + r.recognize_google(audio_text))
            except:
                print("Sorry, I did not get that")


    '''
    get a word2vec of a word
    testing using a pretrained model, can use our own as well
    '''
    def word2vecTest(self):
        print("converting word to vector...")
        from app.main import model
        return str(len(model['cat']))


router = APIRouter()
nlp_engine = NLPEngine()
info_extractor = InfoExtractor()


class InputSpeech(BaseModel):
    text: str

class TaggedSpeech(BaseModel):
    tokens: Optional[str]
    tags: Optional[str]

class Text(BaseModel):
    text: str

class Data(BaseModel):
    user: str


# routes
@router.post("/tag-speech", tags=["nlp"])
async def pos_tag(texts: InputSpeech):
    tags = nlp_engine.generate_pos_tags(text=texts.text)
    return tags


@router.get("/tuples/{text}", tags=["nlp"])
async def info_extract(text: str):
    print("ASDA")
    return info_extractor.extract_tuples(text=text)


@router.get("/test-graph-structure", tags=["nlp"])
async def test_graph():
    return '{"objects": ["sky", "man", "leg", "horse", "tail", "leg","short", "hill", "hill"],"relationships": [[0, "above", 1],[1, "has", 2],[1, "riding", 3],[3, "has", 4],[3, "has", 4],[3, "has", 5]]}'


@router.get("/test-word2vec", tags=["nlp"])
async def test_word2vec(word_to_convert: str):
    return nlp_engine.word2vecTest()


@router.post("/save-example", tags=["nlp"])
async def save_example(array: Text):
    from app.main import model
    from app.main import json_store

    print(array.text)
    array.text = array.text.replace("%22", '"')

    data = json.loads(array.text)
    # node_words = set()
    # for relation in data['array']:
    #     node_words.add(relation['sub'])
    #     node_words.add(relation['obj'])
    #
    # word_embedding_size = 300
    # node_words = list(node_words)
    # node_features = np.empty((len(node_words), word_embedding_size), dtype=float)
    # edge_words = []
    # edge_features = np.empty((len(data['array']), word_embedding_size), dtype=float)
    # edge_direction_truths = np.zeros((len(data['array']), DIRECTION_CLASSES), dtype=float)
    # edge_distance_truths = np.zeros((len(data['array']), DISTANCE_CLASSES), dtype=float)
    # edges_u = []
    # edges_v = []
    # for relation in data['array']:
    #     sub_index = node_words.index(relation['sub'])
    #     obj_index = node_words.index(relation['obj'])
    #
    #     node_features[sub_index] = model[node_words[sub_index]].tolist()
    #     node_features[obj_index] = model[node_words[obj_index]].tolist()
    #
    #     edges_u.append(sub_index)
    #     edges_v.append(obj_index)
    #
    #     edge_words.append(relation['relation'])
    #
    #     edge_direction_truths[len(edge_words)-1][relation['directionTruth']] = 1
    #     edge_distance_truths[len(edge_words)-1][relation['distanceTruth']] = 1
    #
    #     for words in edge_words:
    #         embedding = None
    #         for word in words.split():
    #             if embedding is None:
    #                 embedding = np.array(model[word])
    #             else:
    #                 embedding += np.array(model[word])
    #         edge_features[len(edge_words)-1] = embedding.tolist()
    #
    # example = {'node_words': node_words, 'node_features': node_features.tolist(), 'edge_features': edge_features.tolist(),
    #            'edges_u': edges_u, 'edges_v': edges_v, 'edge_direction_truths': edge_direction_truths.tolist(),
    #            'edge_distance_truths': edge_distance_truths.tolist()}
    #
    # json_store['array'].append(example)
    # with open("./app/resources/dataset.json", "w") as dataset_write:
    #     json.dump(json_store, dataset_write)
    return json.dumps(data)


def predict_animations(graph):
    from app.main import model

    skip = False
    for m_tuple in graph['array']:
        for t_word in m_tuple['relation'].split():
            if skip:
                break
            for b_word in base_animate_words:
                if model.similarity(t_word, b_word) > 0.8:
                    m_tuple['animate'] = True
                    skip = True
                    break




@router.post("/reevaluate-model", tags=["nlp"])
async def reevaluate_model(graph: Text):
    graph.text = graph.text.replace("%22", '"')

    data = json.loads(graph.text)

    data['']

    print("new class is:", int)


@router.post("/predict", tags=["nlp"])
async def predict(graph: Text):
    # convert to tuples
    tuples = info_extract(graph.text)
    
    # convert to json structure
    json_graph = info_extractor.sentenceToGraphData(tuples)

    # convert to dgl graph


    # predict using saved models (distance, direction)
    # return predictions


class SceneData:
    
    def __init__(self, nw, nf, ew, ef, eu, ev):
        self.node_words = nw
        self.node_features = nf
        self.edge_words = ew
        self.edge_features = ef
        self.edges_u = eu
        self.edges_v = ev
    