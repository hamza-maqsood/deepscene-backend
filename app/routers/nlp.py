from typing import Optional
from fastapi import APIRouter
from pydantic import BaseModel
import nltk
import json
import speech_recognition as sr
from pydantic import BaseModel
from app.nlp.information_extraction import InfoExtractor


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


class Data(BaseModel):
    user: str

@router.post("/save-example", tags=["nlp"])
async def save_example(array: Text):
    # print(type(array))
    print(array.text)
    array.text = array.text.replace("%22", '"')
    # data = json.loads(array)
    # print(data)
    return array