# import nltk
# import speech_recognition as sr


# class NLPEngine:

#     def __init__(self):
#         nltk.download('punkt')
#         nltk.download('averaged_perceptron_tagger')

#     def generate_pos_tags(self,text: str):
#         # tokenize the sentence
#         tokens = nltk.word_tokenize(text)
#         print("Tokens", tokens)
#         # do POS tagging on it
#         tags = nltk.pos_tag(tokens=tokens)
#         print("POS tags", tags)

#     def get_speech_and_convert_to_text(self):
#         r = sr.Recognizer()
#         with sr.Microphone() as source:
#             print("Talk")
#             audio_text = r.listen(source)
#             print("Time over, thanks")
#             # recognize_() method will throw a request error if the API is unreachable, hence using exception handling
#             try:
#                 # using google speech recognition
#                 print("Text: " + r.recognize_google(audio_text))
#             except:
#                 print("Sorry, I did not get that")
