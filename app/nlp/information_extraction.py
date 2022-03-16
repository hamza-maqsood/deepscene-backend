import requests
import json


class InfoExtractor:

    # def __init__(self):
    #     self.properties = {'openie.affinity_probability_cap': 2 / 3, }

    def extract_tuples(self, text: str) -> list:
        tuples = []
        url = 'http://localhost:9000/'
        params = {'properties': '{"annotators": "tokenize,ssplit,pos,lemma,depparse,natlog,openie"}',
                  'openie.affinity_probability_cap': 2 / 3, "openie.triple.strict": "true"}
        # Get information about the sentence from CoreNLP
        r = requests.post(url, data=text, params=params, timeout=60)
        data = json.loads(r.text)
        for sentence in data["sentences"]:
            for triple in sentence["openie"]:
                tokens = dict()
                tokens["sub"] = triple["subject"]
                tokens["obj"] = triple["object"]
                tokens["relation"] = triple["relation"]


                tuples.append(tokens)
        # data = json.dumps(data, sort_keys=True, indent=4, separators=(',', ': '))
        # print(data)
        # tuples.append(data)
        # with StanfordOpenIE(properties=self.properties) as client:
        #     for triple in client.annotate(text):
        #         print('|-', triple)
        #         tuples.append(str(triple))
        return tuples
