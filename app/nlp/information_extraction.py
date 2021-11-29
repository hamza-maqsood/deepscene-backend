from openie import StanfordOpenIE


class InfoExtractor:

    def __init__(self):
        self.properties = {'openie.affinity_probability_cap': 2 / 3, }

    def extract_tuples(self, text: str) -> list:
        tuples = []
        with StanfordOpenIE(properties=self.properties) as client:
            for triple in client.annotate(text):
                print('|-', triple)
                tuples.append(str(triple))
        return tuples
