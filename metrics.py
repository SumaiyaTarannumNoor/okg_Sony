from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import BERTScorer
import MeCab

import torch
from transformers import AutoTokenizer, AutoModel
from bert_score import score as bert_score
from bert_score import BERTScorer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from transformers import logging
logging.set_verbosity_error()
# Load the BERT model
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
bert_model = AutoModel.from_pretrained("bert-base-multilingual-cased")
## Initialize the BERTScorer for multilingual BERT or a Japanese-specific BERT
scorer = BERTScorer(model_type="bert-base-multilingual-cased", lang="ja", device='cuda' if torch.cuda.is_available() else 'cpu')

from transformers import BertTokenizer, BertModel


def evaluate_keywords_against_paragraph2(paragraph, keywords):
    # Join keywords into a string for comparison
    keyword_str = ' '.join(keywords)

    # Calculate BLEU-4 score
    bleu4 = sentence_bleu([paragraph.split()], keyword_str.split(), smoothing_function=SmoothingFunction().method1)

    # Calculate ROUGE-1 score
    rouge = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    rouge1 = rouge.score(paragraph, keyword_str)['rouge1'].fmeasure

    return  bleu4, rouge1


def evaluate_keywords_against_paragraph(paragraph, keywords,  lang="ja"):
    # Initialize MeCab Tokenizer
    tokenizer = MeCab.Tagger("-Owakati")
    
    # Tokenize the paragraph and keywords
    tokenized_paragraph = tokenizer.parse(paragraph).strip().split()
    tokenized_keywords = tokenizer.parse(' '.join(keywords)).strip().split()

    # Calculate BLEU-4 score
    bleu4 = sentence_bleu([tokenized_paragraph], tokenized_keywords, smoothing_function=SmoothingFunction().method1)
    # Assuming 'tokenized_paragraph' and 'tokenized_keywords' are already defined
    bleu2 = sentence_bleu([tokenized_paragraph], tokenized_keywords,
                        weights=(0.5, 0.5),  # Equal weights for 1-gram and 2-gram
                        smoothing_function=SmoothingFunction().method1)


    # Calculate ROUGE-1 score
    rouge = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    rouge1 = rouge.score(' '.join(tokenized_paragraph), ' '.join(tokenized_keywords))['rouge1'].fmeasure

    # Initialize BERTScorer with a model suitable for Japanese
    #scorer = BERTScorer(lang=lang, model_type="cl-tohoku/bert-base-japanese", rescale_with_baseline=True)
    scorer = BERTScorer(lang=lang, model_type="bert-base-multilingual-cased", rescale_with_baseline=False)
    
    # Join keywords into a single string
    keyword_str = ' '.join(keywords)
    
    # Calculate BERTScore (P: Precision, R: Recall, F1: F1-score)
    P, R, F1 = scorer.score([keyword_str], [paragraph])

    return bleu2, bleu4, rouge1, F1.item()

def jaccard_similarity(str1, str2):
    # Initialize MeCab Tokenizer
    #tokenizer = MeCab.Tagger("-Owakati")
    
    # Tokenize the strings
    tokens_a = set(tokenizer.tokenize(str1))
    tokens_b = set(tokenizer.tokenize(str2))
    
    # Calculate Jaccard Similarity
    intersection = tokens_a.intersection(tokens_b)
    union = tokens_a.union(tokens_b)
    
    return float(len(intersection) / len(union)) if len(union) != 0 else 0.0
    

def cosine_similarity_calc(str1, str2):
    # Initialize MeCab Tokenizer
    #tokenizer = MeCab.Tagger("-Owakati")
    
    # Tokenize and encode the strings
    inputs_1 = tokenizer(str1, return_tensors='pt', truncation=True, padding=True)
    #print(str1)
    #print(str(inputs_1))
    inputs_2 = tokenizer(str2, return_tensors='pt', truncation=True, padding=True)
    #print(str2)
    #print(str(inputs_2))
    
    # Get embeddings from BERT model
    with torch.no_grad():
        outputs_1 = bert_model(**inputs_1)
        outputs_2 = bert_model(**inputs_2)
    
    # Get the embeddings for the [CLS] token
    # This represents the pooled output for each sentence
    embedding_1 = outputs_1.last_hidden_state[:, 0, :].squeeze().numpy()
    embedding_2 = outputs_2.last_hidden_state[:, 0, :].squeeze().numpy()

    # Reshape the embeddings to 2D arrays (1, -1) to ensure proper input to cosine_similarity
    embedding_1 = embedding_1.reshape(1, -1)
    embedding_2 = embedding_2.reshape(1, -1)
    
    # Calculate Cosine similarity
    cos_sim = cosine_similarity(embedding_1, embedding_2)
    return cos_sim[0, 0]

def find_most_relevant_keywords(keyword_list, dataframe, keyword_column, traffic_column):
    results = {}
    for keyword in keyword_list:
        max_score = -1
        most_relevant = ""
        estimated_traffic = None
        best_cpc = 0
        search_volume = 0
        competitor_score = 0
        best_jaccard = 0
        best_cosine = 0

        for idx, entry in enumerate(dataframe[keyword_column]):
            # Calculate BERTScore
            P, _, _ = scorer.score([keyword], [entry])
            # Calculate Jaccard Similarity
            jaccard = jaccard_similarity(keyword, entry)
            # Calculate Cosine Similarity
            cosine_sim = cosine_similarity_calc(keyword, entry)
            
            if cosine_sim > max_score and cosine_sim > 0.6:
            #if P[0] > max_score and P[0] > 0.6:
                max_score = P[0]
                most_relevant = entry
                estimated_traffic = dataframe[traffic_column].iloc[idx]
                best_cpc = dataframe['CPC ($)'].iloc[idx]
                search_volume = dataframe['月間検索数'].iloc[idx]
                competitor_score = dataframe['競合性'].iloc[idx]
                best_jaccard = jaccard
                best_cosine = cosine_sim

        results[keyword] = {
            'Most Relevant Keyword': most_relevant,
            'BERTScore': max_score.item(),  # Convert tensor to float
            'Estimated Traffic': estimated_traffic,
            'Search Volume': float(search_volume),
            'Competitor Score': float(competitor_score) if competitor_score != '-' else float(0.0),
            'CPC': float(best_cpc) if best_cpc != '-' else float(3.0),
            'Cosine Similarity': best_cosine,
            'Jaccard Similarity': best_jaccard,
        }
    return results

def find_most_relevant_keywords_para(keyword_list, dataframe, keyword_column, traffic_column, cosine_threshold=0.6):
    results = {}
    
    def process_keyword(keyword):
        max_cosine = -1
        most_relevant = ""
        estimated_traffic = None
        best_cpc = 0
        search_volume = 0
        competitor_score = 0
        best_jaccard = 0

        for idx, entry in enumerate(dataframe[keyword_column]):
            # Calculate Cosine Similarity
            cosine_sim = cosine_similarity_calc(keyword, entry)
            
            # Only proceed if cosine similarity is above the threshold
            if cosine_sim > max_cosine and cosine_sim > cosine_threshold:
                max_cosine = cosine_sim
                most_relevant = entry
                estimated_traffic = dataframe[traffic_column].iloc[idx]
                best_cpc = dataframe['CPC ($)'].iloc[idx]
                search_volume = dataframe['月間検索数'].iloc[idx]
                competitor_score = dataframe['競合性'].iloc[idx]
                best_jaccard = jaccard_similarity(keyword, entry)

        return keyword, {
            'Most Relevant Keyword': most_relevant,
            'Cosine Similarity': max_cosine,
            'Estimated Traffic': estimated_traffic,
            'Search Volume': float(search_volume) if search_volume != '-' else 0.0,
            'Competitor Score': float(competitor_score) if competitor_score != '-' else 0.0,
            'CPC': float(best_cpc) if best_cpc != '-' else 3.0,
            'Jaccard Similarity': best_jaccard,
        }

    with ProcessPoolExecutor() as executor:
        future_to_keyword = {executor.submit(process_keyword, kw): kw for kw in keyword_list}
        for future in future_to_keyword:
            keyword, result = future.result()
            results[keyword] = result
    
    return results

def find_best_match_for_keyword(keyword, dataframe, keyword_column, traffic_column, cosine_threshold=0.6):
    max_cosine = -1
    best_entry = {
        'Most Relevant Keyword': "",
        'Cosine Similarity': max_cosine,
        'Estimated Traffic': None,
        'Search Volume': 0,
        'Competitor Score': 0,
        'CPC': 0,
        'Jaccard Similarity': 0,
    }

    for idx, entry in enumerate(dataframe[keyword_column]):
        # Calculate Cosine Similarity
        cosine_sim = cosine_similarity_calc(keyword, entry)
        
        # Only proceed if cosine similarity is above the threshold
        if cosine_sim > max_cosine and cosine_sim > cosine_threshold:
            max_cosine = cosine_sim
            best_entry = {
                'Most Relevant Keyword': entry,
                'Cosine Similarity': max_cosine,
                'Estimated Traffic': dataframe[traffic_column].iloc[idx],
                'Search Volume': float(dataframe['月間検索数'].iloc[idx]) if dataframe['月間検索数'].iloc[idx] != '-' else 0.0,
                'Competitor Score': float(dataframe['競合性'].iloc[idx]) if dataframe['競合性'].iloc[idx] != '-' else 0.0,
                'CPC': float(dataframe['CPC ($)'].iloc[idx]) if dataframe['CPC ($)'].iloc[idx] != '-' else 3.0,
                'Jaccard Similarity': jaccard_similarity(keyword, entry),
            }

    return best_entry

def find_most_relevant_keywords_para2(keyword_list, dataframe, keyword_column, traffic_column, cosine_threshold=0.6):
    results = {}
    
    with ProcessPoolExecutor() as executor:
        # Submit all tasks to the pool
        future_to_keyword = {executor.submit(find_best_match_for_keyword, kw, dataframe, keyword_column, traffic_column, cosine_threshold): kw for kw in keyword_list}
        
        # Collect results
        for future in future_to_keyword:
            keyword = future_to_keyword[future]
            results[keyword] = future.result()
    
    return results

def update_clicks(df, kw_dict, traffic_column ):
    # if column 'Jacard' not exist, create it
    if 'Jacard' not in df.columns:
        df['Jacard'] = 0
    # if column 'Cosine' not exist, create it
    if 'Cosine' not in df.columns:
        df['Cosine'] = 0
    # if column 'BERT' not exist, create it
    if 'BERT' not in df.columns:
        df['BERT'] = 0
    if 'Search Volume' not in df.columns:
        df['Search Volume'] = 0
    if 'Competitor Score' not in df.columns:
        df['Competitor Score'] = 0
    if 'CPC' not in df.columns:
        df['CPC'] = 0
    
    # Iterate through the DataFrame rows
    for index, row in df.iterrows():
        keyword = row['Keyword']
        # Check if the keyword exists in the dictionary
        if keyword in kw_dict:
            # Update the 'Clicks' column with the 'Estimated Traffic' from the dictionary
            df.at[index, 'Clicks'] = kw_dict[keyword][traffic_column]
            df.at[index, 'Jacard'] = kw_dict[keyword]['Jaccard Similarity']
            df.at[index, 'Cosine'] = kw_dict[keyword]['Cosine Similarity']
            df.at[index, 'BERT'] = kw_dict[keyword]['BERTScore']
            df.at[index, 'Search Volume'] = kw_dict[keyword]['Search Volume']
            df.at[index, 'Competitor Score'] = kw_dict[keyword]['Competitor Score']
            df.at[index, 'CPC'] = kw_dict[keyword]['CPC']

            
    return df

def r_kw_plan (kw_list, intro_string):
    """
    Calculate the cosine similarity between a list of keywords and a product description using TF-IDF.

    Args:
    keywords (list of str): The list of keywords or phrases to compare.
    description (str): The full product description text.

    Returns:
    list of float: The list of cosine similarity scores for each keyword.
    """
    # Create a TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()

    # Combine the list of keywords with the description into a single list where
    # the description is the last element
    texts = kw_list + [intro_string]

    # Fit and transform the texts
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Calculate cosine similarity between each keyword vector and the description vector
    similarity_scores = cosine_similarity(tfidf_matrix[:-1], tfidf_matrix[-1:])

    # Flatten the array of scores and return it as a list
    return similarity_scores.flatten().tolist()

def r_kw_plan_bert (kw_list, intro_string):
    """
    Calculate the cosine similarity between a list of keywords and a product description using BERT embeddings.

    Args:
    kw_list (list of str): The list of keywords or phrases to compare.
    intro_string (str): The full product description text.

    •	Precision (P): Measures how many of the tokens in the candidate text are relevant and correctly matched with the reference text.
	•	Recall (R): Measures how many of the tokens in the reference text are relevant and correctly matched with the candidate text.
	•	F1 Score (F1): The harmonic mean of precision and recall, providing a balanced measure of both.

    •	0.4 - 0.6: Moderate similarity
	•	0.6 - 0.8: High similarity
	•	0.8 - 1.0: Very high similarity


    Returns:
    list of float: The list of cosine similarity scores for each keyword.
    """
    P, R, F1 = bert_score(kw_list, [intro_string] * len(kw_list), lang='ja', verbose=False)
    
    # Convert tensor to list
    similarity_scores = F1.tolist()
    
    return similarity_scores