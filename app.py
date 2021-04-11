from flask import Flask, request
import spacy
from spacy.tokenizer import Tokenizer

app = Flask(__name__)
nlp = spacy.load('ru2')
tokenizer = Tokenizer(nlp.vocab)


@app.route('/anonymize', methods=["POST"])
def anonymoize():
    entities = request.json['entities']
    raw_text = request.json['raw_text']
    doc = nlp(raw_text)
    print([X for X in doc.ents])
    return " ".join(
        map(lambda p: filter_data(p, entities),
            [(X, X.ent_type_) for X in doc]))


def filter_data(tuple_data, entities):
    if (str(tuple_data[1]) in entities and str(tuple_data[1]) != ""):
        return "***"
    return str(tuple_data[0])


if __name__ == '__main__':
    app.run(debug=True)