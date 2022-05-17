import dill
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained("tblard/tf-allocine", use_fast=True)
model = TFAutoModelForSequenceClassification.from_pretrained("tblard/tf-allocine")

if __name__ == '__main__':
    try:
        nlp = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
        dill.dump(nlp, open('predict.pkl', 'wb'))
        print("Model created.")
    except Exception as e:
        print('Model creation failed.')
        print(e)
