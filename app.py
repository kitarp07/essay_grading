from flask import Flask, request, jsonify, render_template
from transformers import BertTokenizer
import torch
from model import BertForRegression, BertForRegressionForAllSets
import pandas as pd
import numpy as np

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForRegression.from_pretrained("bert-base-uncased")
all_model = BertForRegressionForAllSets.from_pretrained("bert-base-uncased")

model.load_state_dict(torch.load("model_multi_output_4.pt", map_location=device))
all_model.load_state_dict(torch.load("model_with_all_sets_2.pt", map_location=device))

all_model.to(device)


score_ranges = {
    1: (2, 12),  # Essay Set 1
    2: (2, 10),  # Essay Set 2
    3: (0, 3),   # Essay Set 3
    4: (0, 3),   # Essay Set 4
    5: (0, 4),   # Essay Set 5
    6: (0, 4),   # Essay Set 6
    7: (0, 30),  # Essay Set 7
    8: (0, 60)   # Essay Set 8
}

def inverse_min_max_scaling(predicted_val, essay_set):
    if essay_set in score_ranges:
        min_value, max_value = score_ranges[essay_set]
        
        if max_value != min_value:  # Avoid division by zero
            scale_up_predicted = predicted_val* (max_value - min_value) + min_value
        else:
            scale_up_predicted = min_value  # If all values were the same, original value is min_value
    else:
        raise ValueError(f"Invalid essay set number: {essay_set}")
    
    return scale_up_predicted


def predict_essay(essay, all_model, tokenizer, device, features):
    all_model.eval()
    inputs = tokenizer(essay, return_tensors='pt', padding=True, truncation=True, max_length=512)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    feature_tensor = torch.tensor(features.values, dtype=torch.float32)
    features = feature_tensor.unsqueeze(0).to(device, dtype=torch.float)
    
    with torch.no_grad():
        overall = all_model(input_ids=input_ids, attention_mask=attention_mask, features=features)
        predicted_overall = overall.squeeze().cpu().numpy()

    return predicted_overall


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/scoring', methods=['GET', 'POST'])
def score_essays():
    essays = []
    scores = []
    predicted_scores = []
    essay_sets = []

    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith('.csv'):
            df = pd.read_csv(file)
            df = df.head(10)
            # Extract and process CSV data
            
            essay_sets = df["essay_set"].tolist()
            features = pd.read_csv("test_features.csv")# Adjust as needed
            features_df = features.head(10)
            essays = features_df["essay"].values.tolist()

            features_2 = pd.read_csv("scaled_test_features.csv").head(10)

            features_2 = features_2[["num_grammatical_errors", "adjacent_sentence_similarity","lexical_repetition_ratio",
             "unique_lemmas_count", "pronoun_count", "avg_sentence_length", "discourse_connective_count",
             "vocabulary_richness", "semantic_relevance", "flesch_reading_ease", "flesch_kincaid_grade",
             "gunning_fog_index", "smog_index", "automated_readability_index", "coleman_liau_index"
             ]]
            

            
            # Generate dummy scores for example
            scores = df[["overall_score", "total_score"]].head(10).to_dict(orient='records')
            
            # Predict scores
            predicted_scores = []
            for i in range(len(essays)):
                essay = essays[i]
                feature = features_2.iloc[i]
                predicted_overall = predict_essay(essay, all_model, tokenizer, device, feature)
                scaled_up_predicted = inverse_min_max_scaling(predicted_overall, essay_sets[i])


                predicted_scores.append({'predicted_score': np.round(scaled_up_predicted).astype(np.int64), 'total_score': scores[i]["total_score"]}) 

    # df = pd.read_csv("test_features.csv")
    # essays = df["essay"].values.tolist()
    # essay_50 = essays[:50]
    
    # df_score = pd.read_csv("test_scores_with_total.csv")
    # scores = df_score[["overall_score", "total_score"]]
    # score_50 = scores[:50]
    # essay_set = df_score["essay_set"].values.tolist()
    
    # df_dict = score_50.to_dict(orient='records')

    # essay_set_50 = essay_set[:50]
    return render_template('scoring.html', essays = essays, scores = scores, essay_sets = essay_sets, predicted_scores=predicted_scores)


@app.route('/predict', methods=['POST'])
def predict():
    essay = request.form.get('essay')
    
    if not essay:
        return jsonify({'error': 'No essay provided'}), 400

    # Tokenize the input text
    inputs = tokenizer(essay, return_tensors='pt', padding=True, truncation=True, max_length=512)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.cat(outputs, dim=1).cpu().numpy().tolist()
    
    # Format the response
     # Format the response
    return jsonify({
        'predictions': predictions
    })

