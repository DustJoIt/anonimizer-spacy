from flask import Flask, request
from natasha import NewsNERTagger, Segmenter, NewsEmbedding, Doc

app = Flask(__name__)
segmenter = Segmenter()
emb = NewsEmbedding()
ner_tagger = NewsNERTagger(emb)


@app.route('/anonymize', methods=["POST"])
def anonymoize():
    entities = request.json['entities']
    raw_text = request.json['raw_text']
    doc = Doc(raw_text)
    doc.segment(segmenter)
    doc.tag_ner(ner_tagger)
    return " ".join(map(lambda p: filter_data(p, entities), doc.sents))


def filter_data(sents, entities):
    text = sents.text
    spans = sents.spans
    for span in spans:
        if (span.type in entities):
            text = "".join([
                text[:span.start], "*" * (span.stop - span.start),
                text[span.stop:]
            ])
    return text


if __name__ == '__main__':
    app.run(debug=True)