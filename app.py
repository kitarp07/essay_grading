from flask import Flask, request, jsonify, render_template
from transformers import BertTokenizer
import torch
from model import BertForRegression, BertForRegressionForAllSets, BertForRegression2, BertForRegression_12, BertForRegression_Source_Dependent
from flask_socketio import SocketIO, emit
import pandas as pd
import numpy as np
import io
from flask_cors import CORS
from feature_extraction import calculate_all_features


app = Flask(__name__)
socketio = SocketIO(app)
CORS(app)


app.config['TEMPLATES_AUTO_RELOAD'] = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForRegression.from_pretrained("bert-base-uncased")
all_model = BertForRegression2.from_pretrained("bert-base-uncased")

model_1_2 = BertForRegression_12.from_pretrained("bert-base-uncased")
model_src_dep = BertForRegression_Source_Dependent.from_pretrained("bert-base-uncased")

model.load_state_dict(torch.load("model_multi_output_4.pt", map_location=device))

checkpoint =  torch.load('model_checkpoint_epoch_14.pth', map_location=device)
all_model.load_state_dict(checkpoint['model_state_dict'])

all_model.to(device)

checkpoint_1_2 =  torch.load('model_1_2/model_checkpoint_1_2_2nd_epoch_20.pt', map_location=device)

model_1_2.load_state_dict(checkpoint_1_2['model_state_dict'])

checkpoint_src_dep = torch.load('model_src_dep/src_model_checkpoint_epoch_19.pt', map_location=device)

model_src_dep.load_state_dict(checkpoint_src_dep['model_state_dict'])


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

essay_prompts = {
    
    '1': '''More and more people use computers, but not everyone agrees that this benefits society. Those who support advances in technology believe that computers have a positive effect on people. They teach hand-eye coordination, give people the ability to learn about faraway places and people, and even allow people to talk online with other people. Others have different ideas. Some experts are concerned that people are spending too much time on their computers and less time exercising, enjoying nature, and interacting with family and friends.
    Write a letter to your local newspaper in which you state your opinion on the effects computers have on people. Persuade the readers to agree with you.'''

    ,'2': ''' "All of us can think of a book that we hope none of our children or any other children have taken off the shelf. But if I have the right to remove that book from the shelf -- that work I abhor -- then you also have exactly the same right and so does everyone else. And then we have no books left on the shelf for any of us." --Katherine Paterson, Author
    Write a persuasive essay to a newspaper reflecting your vies on censorship in libraries. Do you believe that certain materials, such as books, music, movies, magazines, etc., should be removed from the shelves if they are found offensive? Support your position with convincing arguments from your own experience, observations, and/or reading.'''

    #source_essay
    ,'3':  '''Write a response that explains how the features of the setting affect the cyclist. In your response, include examples from the essay that support your conclusion.'''

    #source_essay
    ,'4':  ''' Read the last paragraph of the story. When they come back, Saeng vowed silently to herself, in the spring, when the snows melt and the geese return and this hibiscus is budding, then I will take that test again. Write a response that explains why the author concludes the story with this paragraph. In your response, include details and examples from the story that support your ideas.'''

    #source_essay
    ,'5': '''Describe the mood created by the author in the memoir. Support your answer with relevant and specific information from the memoir.'''

    ##source_essay
    ,'6':  '''Based on the excerpt, describe the obstacles the builders of the Empire State Building faced in attempting to allow dirigibles to dock there. Support your answer with relevant and specific information from the excerpt.'''

    ,'7':  '''Write about patience. Being patient means that you are understanding and tolerant. A patient person experience difficulties without complaining.
         Do only one of the following: write a story about a time when you were patient OR write a story about a time when someone you know was patient OR write a story in your own way about patience.'''

    ,'8': '''We all understand the benefits of laughter. For example, someone once said, “Laughter is the shortest distance between two people.” Many other people believe that laughter is an important part of any relationship. Tell a true story in which laughter was one element or part. '''

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

            features_2 = features_2[[
                            "num_grammatical_errors", "adjacent_sentence_similarity", "lexical_repetition_ratio",
                            "unique_lemmas_count", "pronoun_count", "avg_sentence_length", "discourse_connective_count",
                            "vocabulary_richness", "semantic_relevance", "flesch_reading_ease", "flesch_kincaid_grade","gunning_fog_index",
                            "smog_index","automated_readability_index","coleman_liau_index"
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

# @app.route('/scoring', methods=['GET', 'POST'])
# def score_essays():
#     essays = []
#     scores = []
#     predicted_scores = []
#     essay_sets = []

#     if request.method == 'POST':
#         file = request.files['file']
#         if file and file.filename.endswith('.csv'):
#             df = pd.read_csv(file)
#             df = df.head(10)
#             # Extract and process CSV data
            
#             essay_sets = df["essay_set"].tolist()
#             features = pd.read_csv("test_features.csv")# Adjust as needed
#             features_df = features.head(10)
#             essays = features_df["essay"].values.tolist()

#             features_2 = pd.read_csv("scaled_test_features.csv").head(10)

#             features_2 = features_2[[
#                             "num_grammatical_errors", "adjacent_sentence_similarity", "lexical_repetition_ratio",
#                             "unique_lemmas_count", "pronoun_count", "avg_sentence_length", "discourse_connective_count",
#                             "vocabulary_richness", "semantic_relevance", "flesch_reading_ease", "flesch_kincaid_grade","gunning_fog_index",
#                             "smog_index","automated_readability_index","coleman_liau_index"
#                         ]]
            

            
#             # Generate dummy scores for example
#             scores = df[["overall_score", "total_score"]].head(10).to_dict(orient='records')
            
#             # Predict scores
#             predicted_scores = []
#             for i in range(len(essays)):
#                 essay = essays[i]
#                 feature = features_2.iloc[i]
#                 predicted_overall = predict_essay(essay, all_model, tokenizer, device, feature)
#                 scaled_up_predicted = inverse_min_max_scaling(predicted_overall, essay_sets[i])


#                 predicted_scores.append({'predicted_score': np.round(scaled_up_predicted).astype(np.int64), 'total_score': scores[i]["total_score"]}) 

#     return render_template('scoring.html', essays = essays, scores = scores, essay_sets = essay_sets, predicted_scores=predicted_scores)

@app.route('/score')
def score():
    return render_template('score.html')

@app.route('/gradingviz')
def viz():
    return render_template('viz.html')

@socketio.on('process_essays')
def handle_process_essays(data):
    csv_file = data['csv_file']
    # Read the CSV file into a DataFrame
    csv_file_str = csv_file.decode('utf-8')

    df = pd.read_csv(io.StringIO(csv_file_str))

    df = df.head(10)
    
    essay_sets = df["essay_set"].tolist()
    actual_scores = df["overall_score"].tolist()
    total_scores = df["total_score"].tolist()
    
    essays = df["essay"].values.tolist()  # Adjust to your actual column name

    features_2 = df[[
            'num_grammatical_errors', 'lexical_repetition_ratio',
            'unique_lemmas_count', 'avg_sentence_length',
            'discourse_connective_count', 'vocabulary_richness',
            'semantic_relevance', 'flesch_reading_ease', 'flesch_kincaid_grade'
            ]]

    for i in range(len(essays)):
        essay = essays[i]
        essay_set = essay_sets[i]
        actual_score = actual_scores[i]
        total_score = total_scores[i]
        feature = features_2.iloc[i]
        predicted_overall = predict_essay(essay, all_model, tokenizer, device, feature)
        score = inverse_min_max_scaling(predicted_overall, essay_sets[i])
        socketio.emit('score_update', {'essay': essay, 'score': score, 'essay_set': essay_set, 'actual_score':actual_score, 'total_score': total_score }) 


@app.route('/predict-1-2-traits', methods=['POST'])
def predict_traits():
    data = request.get_json()
    
    essay_type = data.get('essayType')
    essay_set = data.get('essaySet')
    essay_text = data.get('essayText')

    prompt = essay_prompts.get(essay_set)

    feature_list = calculate_all_features(essay_text, prompt)

    feature_tensor = torch.tensor(feature_list, dtype=torch.float32)
    features = feature_tensor.unsqueeze(0).to(device, dtype=torch.float)
    # Tokenize the input text
    inputs = tokenizer(essay_text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    if essay_set in ['1','2']:
        # Make prediction
        with torch.no_grad():
            outputs = model_1_2(input_ids=input_ids, attention_mask=attention_mask, features = features)
            predictions = torch.cat(outputs, dim=1).cpu().numpy().tolist()
        # Format the response
        # Format the response
            pred_list = predictions[0]
            modified_list = [max(0, val) for val in pred_list] 
            rubric_scores = [
                    {"name": "Content", "score": np.rint(modified_list[0])},
                    {"name": "Organization", "score": np.rint(modified_list[1])},
                    {"name": "Word Choice", "score": np.rint(modified_list[2])},
                    {"name": "Sentence Fluency", "score": np.rint(modified_list[3])},
                    {"name": "Conventions", "score": np.rint(modified_list[4])}
                ]
    else:
        with torch.no_grad():
            outputs = model_1_2(input_ids=input_ids, attention_mask=attention_mask, features = features)
            predictions = torch.cat(outputs, dim=1).cpu().numpy().tolist()
            pred_list = predictions[0]
            modified_list = [max(0, val) for val in pred_list] 
            rubric_scores = [
                    {"name": "Content", "score": np.rint(modified_list[0])},
                    {"name": "Prompt Adherence", "score": np.rint(modified_list[1])},
                    {"name": "Language", "score": np.rint(modified_list[2])},
                    {"name": "Narrativity", "score": np.rint(modified_list[3])},      
                ]
    # Format the response
    return jsonify({
        'message': 'Essay scored successfully!',
        'scores': rubric_scores
    })

if __name__ == '__main__':
    socketio.run(app)

