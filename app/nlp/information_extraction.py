from openie import StanfordOpenIE
from monkeylearn import MonkeyLearn

class InfoExtractor:

    def __init__(self):
        self.properties = {'openie.affinity_probability_cap': 2 / 3, }

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
        with StanfordOpenIE(properties=self.properties) as client:
            for triple in client.annotate(text):
                print('|-', triple)
                triple["placement"] = self.identify_placement(text)
                print('|-', triple)
                tuples.append(str(triple).replace("\'","\""))
        return tuples