import language_tool_python
import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel
from collections import Counter
import nltk

# nltk.download('punkt')
# nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import torch
import textstat




nlp = spacy.load("en_core_web_sm")
calc_embedder = pipeline("feature-extraction", model="bert-base-uncased", tokenizer="bert-base-uncased", truncation= True)
calc_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
calc_model = AutoModel.from_pretrained("bert-base-uncased")

m2_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
m2_model = BertModel.from_pretrained('bert-base-uncased')

tool = language_tool_python.LanguageTool('en-US')

def get_grammar_spelling_errors(text):
    matches = tool.check(text)
    return matches

def get_sentence_embeddings(text, max_length=512):
    # Tokenize the text into sentences
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]

    embeddings = []
    for sentence in sentences:
        tokens = calc_tokenizer(sentence, return_tensors='pt', truncation=True, max_length=max_length)
        with torch.no_grad():
            outputs = calc_model(**tokens)
            sentence_embedding = outputs.last_hidden_state.mean(dim=1).numpy().flatten()
        embeddings.append(sentence_embedding)

    return embeddings

def sentence_similarity(text):
    # Get embeddings for the text
    embeddings = get_sentence_embeddings(text)

    # Calculate cosine similarity between consecutive sentence embeddings
    similarities = [cosine_similarity([embeddings[i]], [embeddings[i+1]])[0, 0] for i in range(len(embeddings) - 1)]
    avg_similarity = np.mean(similarities) if similarities else 0

    return avg_similarity

def extract_features(text):
    doc = nlp(text)

    # Lexical Cohesion Features
    tokens = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct]
    lemmas = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
    lemma_counts = Counter(lemmas)

    lexical_repetition_ratio = sum(count for count in lemma_counts.values() if count > 1) / len(tokens) if tokens else 0
    unique_lemmas_count = len(lemma_counts)

    # Pronoun Usage
    pronouns = [token.text.lower() for token in doc if token.pos_ == "PRON"]
    pronoun_count = len(pronouns)

    # Conjunctions and Connectives
    conjunctions = {"and", "or", "but", "because", "although", "however", "therefore", "moreover", "thus", "furthermore", "then"}
    conjunction_count = sum(1 for token in doc if token.text.lower() in conjunctions)

    # Entity Coherence
    entities = [ent.text.lower() for ent in doc.ents]
    entity_counts = Counter(entities)
    entity_repetition_count = sum(count for count in entity_counts.values() if count > 1)

    # Sentence-Level Features
    sentence_lengths = [len(sent) for sent in doc.sents]
    avg_sentence_length = np.mean(sentence_lengths) if sentence_lengths else 0
    sentence_length_std = np.std(sentence_lengths) if sentence_lengths else 0

    # Paragraph-Level Features
    paragraphs = text.split('\n\n')
    paragraph_lengths = [len(p.split()) for p in paragraphs]
    avg_paragraph_length = np.mean(paragraph_lengths) if paragraph_lengths else 0

    features = {
        "lexical_repetition_ratio": lexical_repetition_ratio,
        "unique_lemmas_count": unique_lemmas_count,
        "pronoun_count": pronoun_count,
        "conjunction_count": conjunction_count,
        "entity_repetition_count": entity_repetition_count,
        "avg_sentence_length": avg_sentence_length,
        "sentence_length_std": sentence_length_std,
        "avg_paragraph_length": avg_paragraph_length
    }

    return features

# Download NLTK resources (if not already downloaded)



def count_discourse_connectives(essay):
    """
    Count the occurrences of discourse connectives in a given essay.

    Parameters:
    - essay (str): The text of the essay.
    - discourse_connectives (list): List of discourse connectives to look for.

    Returns:
    - connective_counts (Counter): A Counter object with the count of each connective.
    - total_count (int): Total count of discourse connectives found.
    """

    # Define discourse connectives
    discourse_connectives = [
        "furthermore", "moreover", "in addition", "besides", "also", "similarly",
        "as well as", "equally important", "not only", "but also", "however",
        "although", "though", "on the other hand", "nevertheless", "nonetheless",
        "whereas", "while", "despite", "in contrast", "alternatively", "therefore",
        "consequently", "thus", "as a result", "because", "since", "due to",
        "hence", "if", "unless", "provided that", "in case", "as long as",
        "on the condition that", "then", "subsequently", "earlier", "later",
        "meanwhile", "before", "after", "during", "once", "until", "when",
        "indeed", "in fact", "certainly", "especially", "particularly",
        "importantly", "in conclusion", "to sum up", "overall", "in summary",
        "thus", "consequently", "ultimately", "for example", "for instance",
        "such as", "to illustrate", "namely", "in other words", "that is to say",
        "to put it another way"
    ]

    # Tokenize the essay
    tokens = word_tokenize(essay.lower())

    # Count occurrences of discourse connectives
    connective_counts = Counter(token for token in tokens if token in discourse_connectives)
    total_count = sum(connective_counts.values())

    return connective_counts, total_count


def calculate_vocabulary_richness(text):
    """
    Calculate the Type-Token Ratio (TTR) to measure vocabulary richness in a given text.

    Parameters:
    text (str): The input text for which to calculate vocabulary richness.

    Returns:
    float: The Type-Token Ratio (TTR) representing the vocabulary richness.
    """
    # Tokenize the text
    tokens = word_tokenize(text.lower())

    # Remove punctuation and stopwords
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Calculate the number of unique words and total words
    unique_words = set(tokens)
    total_words = len(tokens)
    unique_word_count = len(unique_words)

    # Calculate Type-Token Ratio (TTR)
    if total_words > 0:
        ttr = unique_word_count / total_words
    else:
        ttr = 0.0

    return ttr

def get_embedding(text, model, tokenizer):
    # Tokenize text
    inputs = tokenizer(text, return_tensors='pt', truncation=False, padding=False)
    input_ids = inputs['input_ids'].squeeze()
    attention_mask = inputs['attention_mask'].squeeze()

    # Process text in chunks
    chunk_size = 512
    embeddings = []
    for i in range(0, len(input_ids), chunk_size):
        chunk_ids = input_ids[i:i + chunk_size].unsqueeze(0)
        chunk_attention_mask = attention_mask[i:i + chunk_size].unsqueeze(0)
        with torch.no_grad():
            outputs = model(input_ids=chunk_ids, attention_mask=chunk_attention_mask)
        chunk_embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        embeddings.append(chunk_embedding)

    # Aggregate chunk embeddings
    aggregated_embedding = np.mean(np.vstack(embeddings), axis=0)
    return aggregated_embedding

def calculate_semantic_relevance(prompt, essay):
    prompt_embedding = get_embedding(prompt, m2_model, m2_tokenizer)
    essay_embedding = get_embedding(essay, m2_model, m2_tokenizer)

    prompt_embedding = prompt_embedding.reshape(1, -1)
    essay_embedding = essay_embedding.reshape(1, -1)

    similarity = cosine_similarity(prompt_embedding, essay_embedding)
    return similarity[0, 0]

def calculate_all_features(prompt,essay):
  num_grammatical_errors = len(get_grammar_spelling_errors(essay))
  features = extract_features(essay)
  connective_counts, total_count = count_discourse_connectives(essay)
  vocabulary_richness = calculate_vocabulary_richness(essay)
  semantic_relevance = calculate_semantic_relevance(prompt, essay)
  #readability scores
  flesch_reading_ease = textstat.flesch_reading_ease(essay)
  flesch_kincaid_grade = textstat.flesch_kincaid_grade(essay)

  feature_list = [
    num_grammatical_errors,
    features["lexical_repetition_ratio"],
    features["unique_lemmas_count"],
    features["avg_sentence_length"],
    total_count,  # discourse connective count
    vocabulary_richness,
    semantic_relevance,
    flesch_reading_ease,
    flesch_kincaid_grade
    ]

  return feature_list

