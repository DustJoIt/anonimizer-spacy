from flask import Flask, request
from natasha import NewsNERTagger, Segmenter, NewsEmbedding, Doc, MorphVocab, DatesExtractor

app = Flask(__name__,
            static_url_path='',
            static_folder='build/')

# ner
segmenter = Segmenter()
emb = NewsEmbedding()
ner_tagger = NewsNERTagger(emb)

# for dates extracting
morph_vocab = MorphVocab()
dates_extractor = DatesExtractor(morph_vocab)


@app.route('/anonymize', methods=["POST"])
def anonymoize():
    entities = request.json['entities']
    raw_text = request.json['raw_text']
    if "DATE" in entities:
        raw_text = anonymize_date(raw_text)
    doc = Doc(raw_text)
    doc.segment(segmenter)
    doc.tag_ner(ner_tagger)
    return filter_data(doc.spans, raw_text, entities)


def anonymize_data(span, text):
    return "".join(
        [text[:span.start], "*" * (span.stop - span.start), text[span.stop:]])


def filter_data(spans, raw_text, entities):
    for span in spans:
        if (span.type in entities):
            raw_text = anonymize_data(span, raw_text)
    return raw_text


def anonymize_date(text):
    for span in list(dates_extractor(text)):
        text = anonymize_data(span, text)
    return text


if __name__ == '__main__':
    app.run(debug=True)
