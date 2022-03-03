from monkeylearn import MonkeyLearn
import requests
import json

class InfoExtractor:

    # def __init__(self):
    #     self.properties = {'openie.affinity_probability_cap': 2 / 3, }

    def identify_placement(self, text: str):
    #        # sample call: self.identify_placement("a car below the bird")
        ml = MonkeyLearn('7324b388540993dbe387dc182b5834f2faac441b')
        data = [text]
        model_id = 'cl_Gcg7fkeo'
        response = ml.classifiers.classify(model_id, data)
        # print(response.body)
        return response.body[0]['classifications'][0]['tag_name']
    
    def extract_tuples(self, text: str) -> list:
        tuples = []

        url = 'https://tranquil-garden-18140.herokuapp.com:9000/'
        params = {'properties': '{"annotators": "tokenize,ssplit,pos,lemma,depparse,natlog,openie"}',
                  'openie.affinity_probability_cap': 2 / 3, "openie.triple.strict": "true"}
        # Get information about the sentence from CoreNLP
        r = requests.post(url, data=text, params=params, timeout=60)
        data = json.loads(r.text)
        for sentence in data["sentences"]:
            for triple in sentence["openie"]:
                tokens = dict()
                tokens["subject"] = triple["subject"]
                tokens["relation"] = triple["relation"]
                tokens["object"] = triple["object"]
                tuples.append(tokens)
        # data = json.dumps(data, sort_keys=True, indent=4, separators=(',', ': '))
        # print(data)
        # tuples.append(data)
        # with StanfordOpenIE(properties=self.properties) as client:
        #     for triple in client.annotate(text):
        #         print('|-', triple)
        #         tuples.append(str(triple))
        return tuples
