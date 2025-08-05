#imports
from pypdf import PdfReader
from docx import Document
import spacy
import re

#file upload from user
#pdf function
def get_pdf_text(pdf_path):
  text = ""
  try:
    with open(pdf_path, 'rb') as file:
      reader = PdfReader(file)
      for page in reader.pages:
        text += page.extract_text()
  except Exception as e:
    print(f"Error reading your PDF file: {e}")
  return text

#docx function
def get_docx_text(docx_path):
  text = ""
  try:
    document = Document(docx_path)
    for paragraph in document.paragraphs:
      text += paragraph.text + "\n"
  except Exception as e:
    print(f"Error reading you Docx file: {e}")
  return text

#en_core_web_sm is the english language model
nlp = spacy.load('en_core_web_sm')
stopwords = nlp.Defaults.stop_words

#normalize text (lowercase, no urls, special characters, stopwords)
def process(text):
  text = text.lower()
  url = re.compile(r'https?://\S+|www\.\S+')
  email = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
  clean_text = url.sub('', text)
  clean_text = email.sub('', clean_text)
  #only words and whitespace
  clean_text = re.sub('[^a-zA-Z]', ' ', text)
  #no stop words
  stop_words = stopwords
  clean_text = ' '.join(word for word in clean_text.split() if word.lower() not in stop_words)

  return clean_text

def get_keyphrases(text):
  key_phrases = []
  doc = nlp(text)
  for chunk in doc.noun_chunks:
    key_phrases.append(chunk.text)
  return key_phrases

def get_keywords(text):
  result = []
  pos_tag = ['PROPN', 'ADJ', 'NOUN']
  doc = nlp(text)
  for token in doc:
    #if(token.text in nlp.Defaults.stop_words or token.text in punctuation):
      #continue
    if(token.pos_ in pos_tag):
      result.append(token.text)
  return result
