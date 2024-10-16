import spacy
from spacy import displacy

# Load trained NER model
nlp_ner = spacy.load("model-best")


# Test the model with a sample text
def extract_mountain_names(text):
    doc = nlp_ner(text)
    colors = {"MOUNTAIN_NAME": "#668eab"}
    options = {"colors": colors}
    displacy.render(doc, options=options, style="ent", jupyter=True)
    return [ent.text for ent in doc.ents if ent.label_ == "MOUNTAIN_NAME"]


# Example usage
text = "Mount Everest, known as the tallest mountain in the world, and K2, the second highest peak, are both part of " \
       "the majestic Himalayas. The Rocky Mountains in North America are home to stunning peaks like Mount Elbert and " \
       "Mount Rainier, attracting hikers and adventurers alike. In the Andes mountain range, Ojos del Salado, " \
       "the highest active volcano in the world, stands proudly alongside the picturesque Fitz Roy. Meanwhile, " \
       "Mont Blanc, the highest mountain in the Alps, and Matterhorn, with its iconic pyramid shape, draw countless " \
       "climbers each year to experience their beauty. Finally, the Atlas Mountains in Morocco feature Toubkal, " \
       "the highest peak in North Africa, surrounded by breathtaking landscapes and rich cultural heritage. "
mountain_names = extract_mountain_names(text)
print(mountain_names)
