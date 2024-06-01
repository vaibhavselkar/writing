import pandas as pd
import re
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker
import spacy
from tabulate import tabulate
from gramformer import Gramformer
import torch
from IPython.display import display, Markdown

nlp = spacy.load("en_core_web_sm")

# Load CEFR vocabulary
cefr_vocab = pd.read_csv('./cefr-vocab-cefrj-octanove.csv')
cefr_dict = {k: v for k, v in cefr_vocab[['headword', 'CEFR']].values}
word_set = set(cefr_vocab.headword)

grammar_fullforms = {'ADV': 'Adverb', 'PREP': 'Prepositions', 'PRON': 'Pronoun', 'WO': 'Wrong Order', 'VERB': 'Verbs',
                    'VERB:SVA': 'Singular-Plural', 'VERB:TENSE': 'Verb Tenses', 'VERB:FORM': 'Verb Forms',
                    'VERB:INFL': 'Verbs', 'SPELL': 'Spelling', 'OTHER': 'Other', 'NOUN': 'Other', 'NOUN:NUM': 'Singular-Plural',
                    'DET': 'Articles', 'MORPH': 'Other', 'ADJ': 'Adjectives', 'PART': 'Other', 'ORTH': 'Other',
                    'CONJ': 'Conjugations', 'PUNCT': 'Punctuation'}

lemmatizer = WordNetLemmatizer()
spell = SpellChecker()

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(1212)
gf = Gramformer(models=1, use_gpu=False)

def strikethrough(text):
    return ''.join([c + '\u0336' for c in text])

def misspelled_words(input_text):
    word_list = [a.lower() for a in word_tokenize(re.sub('\W+', ' ', input_text))]
    lemma_words = [lemmatizer.lemmatize(w) for w in word_list]
    return spell.unknown(lemma_words)

def sentence_spelling_correction(input_sentence):
    word_list = [a.lower() for a in word_tokenize(re.sub('\W+', ' ', input_sentence))]
    lemma_words = [lemmatizer.lemmatize(w) for w in word_list]
    corrected_words = [spell.correction(word) if len(spell.unknown([word])) else word for word in lemma_words]
    return " ".join(corrected_words) + "."

def text_grammar_correction(input_text):
    sentences = sent_tokenize(input_text)
    edits = []
    corrected_text = ''
    color_corrected_text = ''
    for sentence in sentences:
        corrected_sentences = gf.correct(sentence, max_candidates=1)
        for corrected_sentence in corrected_sentences:
            all_edits = gf.get_edits(sentence, corrected_sentence)
            if len(all_edits):
                edits += [a[0] for a in all_edits]
                orig = re.split(' ', sentence)
                amend = re.split(' ', corrected_sentence)
                amend_plus = []
                start = 0
                for edit in all_edits:
                    amend_plus.extend(orig[start:edit[2]])
                    if len(edit[1]):
                        amend_plus.extend(['<span style="background-color:#ffffff;color:#ff3f33">' + strikethrough(edit[1]) + '</span>'])
                    if len(edit[4]):
                        amend_plus.extend(['<span style="color:#07b81a">' + edit[4] + '</span>'])
                    start = edit[3]
                amend_plus.extend(orig[edit[3]:])
                color_corrected_sentence = ' '.join(amend_plus)
                corrected_text += ' ' + corrected_sentence
                color_corrected_text += ' ' + color_corrected_sentence
            else:
                corrected_text += ' ' + sentence
                color_corrected_text += ' ' + sentence            
    mistake_stats = pd.Series([grammar_fullforms[a] for a in edits]).value_counts()
    return corrected_text.strip(), color_corrected_text.strip(), edits, mistake_stats

def cefr_ratings(input_text):
    nopunc_input_text = re.sub(r'[^\w\s]', '', input_text.lower())
    nopunc_input_text = re.sub(r'[0-9]', '', nopunc_input_text)
    words = word_tokenize(nopunc_input_text)
    lemma_words = [lemmatizer.lemmatize(word.lower()) for word in words]

    pos_values = ['v', 'a', 'n', 'r', 's']
    cefr_list = []
    cefr_mapping = {}
    for word in lemma_words:
        if word in word_set:
            cefr_list.append(cefr_dict[word])
            cefr_mapping[word] = cefr_dict[word]
        else:      
            for pos_value in pos_values:
                changed_word = lemmatizer.lemmatize(word, pos=pos_value)
                if changed_word != word:
                    break
            if changed_word in word_set:
                cefr_list.append(cefr_dict[changed_word])
                cefr_mapping[changed_word] = cefr_dict[changed_word]
            else:
                cefr_list.append('uncategorized')
                cefr_mapping[word] = 'uncategorized'
    return cefr_mapping

def process_text(input_text):
    corrected_text, color_corrected_text, edits, mistake_stats = text_grammar_correction(input_text)
    
    # Collecting results in a dictionary for structured output
    result = {
        "corrected_text": corrected_text,
        "color_corrected_text": color_corrected_text,
        "mistake_statistics": mistake_stats.to_dict(),
        "cefr_ratings": {},
        "uncategorized_words": []
    }
    
    # Calculating CEFR ratings
    cefr_mapping = cefr_ratings(corrected_text)
    result["cefr_ratings"] = pd.Series(cefr_mapping.values()).value_counts().to_dict()
    
    # Collecting uncategorized words
    result["uncategorized_words"] = [word for word in cefr_mapping.keys() if cefr_mapping[word] == 'uncategorized']
    
    return result

if __name__ == "__main__":
    input_text = input("Enter the text to be processed: ")
    result = process_text(input_text)
    
    # Displaying the result in a user-friendly manner
    print("Corrected Text:\n", result["corrected_text"])
    display(Markdown(result["color_corrected_text"]))
    
    print("\nMistake Statistics:")
    print(tabulate(pd.Series(result["mistake_statistics"]).reset_index(), headers=["Mistake", "Count"], tablefmt='fancy_grid'))
    
    print("\nCEFR Ratings:")
    print(tabulate(pd.Series(result["cefr_ratings"]).reset_index(), headers=["CEFR Level", "Count"], tablefmt='fancy_grid'))
    
    print("\nUncategorized Words:", result["uncategorized_words"])
