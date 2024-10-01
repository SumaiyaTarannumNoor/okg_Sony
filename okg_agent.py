
import configparser
import os

from langchain import hub
from langchain_community.document_loaders import TextLoader, CSVLoader, DataFrameLoader
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import AzureChatOpenAI
from langchain.agents import initialize_agent, AgentType, AgentExecutor, create_react_agent
# from langchain import SerpAPIWrapper
from langchain_community.utilities import SerpAPIWrapper

import MeCab
import pandas as pd
from datetime import datetime, timedelta

import re

from metrics import evaluate_keywords_against_paragraph, jaccard_similarity, cosine_similarity_calc, find_most_relevant_keywords,find_best_match_for_keyword,update_clicks,r_kw_plan_bert,r_kw_plan


from concurrent.futures import ProcessPoolExecutor

import torch
from transformers import AutoTokenizer, AutoModel
from bert_score import score as bert_score
from bert_score import BERTScorer
from sklearn.feature_extraction.text import CountVectorizer

from transformers import logging
logging.set_verbosity_error()

# Load the BERT model
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
bert_model = AutoModel.from_pretrained("bert-base-multilingual-cased")
## Initialize the BERTScorer for multilingual BERT or a Japanese-specific BERT
scorer = BERTScorer(model_type="bert-base-multilingual-cased", lang="ja", device='cuda' if torch.cuda.is_available() else 'cpu')

from transformers import BertTokenizer, BertModel
#tokenizer = BertTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
#bert_model = BertModel.from_pretrained("cl-tohoku/bert-base-japanese")
#scorer = BERTScorer(lang="ja", device='cuda' if torch.cuda.is_available() else 'cpu')
#scorer = BERTScorer(model_type="cl-tohoku/bert-base-japanese", lang="ja", device='cuda' if torch.cuda.is_available() else 'cpu')


from langchain.agents import load_tools

from load_and_embed import custermized_trend_retriever, custermized_retriever
from utils import concatenate_llm_parts, concatenate_reflection_beginning,run_with_retries




class okg_agent:
    def __init__(self ,config_file = './config_3_day_obs.ini'):

        # 0. Read the configuration file
        self.config = configparser.ConfigParser()
        try:        
            self.config.read(config_file)
            #self.config.read('./config_base.ini')
        except Exception as e:
            raise ValueError("Failed to read the configuration file: " + str(e))
        
        self.observation_period = int(self.config['SYSTEM']['OBSERVATION_PERIOD'])
        
        self.csv_file_path = self.config['FILE']['CSV_FILE']

        self.setting_day = pd.to_datetime (self.config['SYSTEM']['SETTING_DAY'])

        self.dataframe = pd.read_csv(str(self.config['FILE']['CSV_FILE']))
    

        if str(self.config['CAMPAIGN']['PRODUCT_NAME']) == 'ソニーテレビ ブラビア':
            self.df_score = pd.read_csv('./preprocessing/data/score_data/rakkokeyword_sony_bravia.csv', delimiter='\t', quotechar='"', encoding='utf-16')
        elif str(self.config['CAMPAIGN']['PRODUCT_NAME']) == 'ソニー損保 医療保険':
            self.df_score = pd.read_csv('./preprocessing/data/score_data/rakkokeyword_sony_insurance.csv', delimiter='\t', quotechar='"', encoding='utf-16')
        elif str(self.config['CAMPAIGN']['PRODUCT_NAME']) == 'ソニーデジタル一眼カメラ α（アルファ）':
            self.df_score = pd.read_csv('./preprocessing/data/score_data/rakkokeyword_sony_camera.csv', delimiter='\t', quotechar='"', encoding='utf-16')
        elif str(self.config['CAMPAIGN']['PRODUCT_NAME']) == 'ソニー銀行 住宅ローン':
            self.df_score = pd.read_csv('./preprocessing/data/score_data/rakkokeyword_sony_bank_morgage.csv', delimiter='\t', quotechar='"', encoding='utf-16')
        elif str(self.config['CAMPAIGN']['PRODUCT_NAME']) == 'ソニー Prediction One':
            self.df_score = pd.read_csv('./dataset/sony_prediction_one.csv', delimiter='\t')
        
        else:
            raise ValueError("Failed to read the PRODUCT_NAME: " + str(self.config['CAMPAIGN']['PRODUCT_NAME']))
        
        # only keep the first 130 rows of df_score
        self.df_score = self.df_score.iloc[:130]

        os.environ['SERPAPI_API_KEY'] = self.config['KEY']['SERPAPI_API_KEY']
    
    def run(self):
        
        #setting_day = pd.to_datetime (self.config['SYSTEM']['SETTING_DAY'])
        #setting_day = datetime.now()
        #observation_period = int(self.config['SYSTEM']['OBSERVATION_PERIOD'])

        # for lambda debug
        # Comment out when real implementation
        
        step = 0

        good_kw_list = []   

        # get reference dataphrame

        
        # read dataframe from csv   
        #df = pd.read_csv(self.csv_file_path)
        df = self.dataframe
        # rejected_kw_list = rejected_kw_list
        # Read the list from the file
       
        with open('./preprocessing/data/string_list.txt', 'r') as file:
            rejected_kw_list = [line.strip() for line in file]
        # ignore the last two rows  
        #df = df.iloc[:-2]
        # select the columns
        df = df[['Keyword', 'Match type', 'Category', 'Clicks']]

        # define an empty dict to output, the key is as same as click dict
        out_key_word_dict = []
        # setting the keyword for every 4 days, if setting day is not 4th, 8th, 12th, 16th, 20th, 24th, 28th,  return the empty dict
        if self.setting_day.day not in [4, 8, 12, 16, 20, 24, 28]:
            return out_key_word_dict
      
        
        # only keep the rows with the Match type is 'Phrase match'
        df = df[df['Match type'] == 'Phrase match']
        # remove "" from the 'Keyword' column
        df['Keyword'] = df['Keyword'].str.replace('"', '')
        # remove the colomn of 'Match type'
        df = df.drop(columns=['Match type'])
        # save it to a new csv file
        df.to_csv('./current_KW.csv', index=False)

        #KW_loader = CSVLoader('./current_KW.csv')
        
        mean_score = 0
        mean_jacard_score = 0
        mean_cosine_score = 0
        mean_bert_score = 0
        mean_search_volume = 0
        mean_competitor_score = 0
        mean_cpc = 0

        while True:
        #if str(self.config['REFLECTION']['FLAG']) == 'False':
            if step == 0:

                if str(self.config['CAMPAIGN']['PRODUCT_NAME']) == 'ソニーテレビ ブラビア':
                    KW_loader = CSVLoader('./preprocessing/data/kw_data/initial_KW_sony_bravia.csv')
                    df = pd.read_csv('./preprocessing/data/kw_data/initial_KW_sony_bravia.csv')
                elif str(self.config['CAMPAIGN']['PRODUCT_NAME']) == 'ソニー損保 医療保険':
                    KW_loader = CSVLoader('./preprocessing/data/kw_data/initial_KW_sony_insurance.csv')
                    df = pd.read_csv('./preprocessing/data/kw_data/initial_KW_sony_insurance.csv')
                elif str(self.config['CAMPAIGN']['PRODUCT_NAME']) == 'ソニーデジタル一眼カメラ α（アルファ）':
                    KW_loader = CSVLoader('./preprocessing/data/kw_data/initial_KW_sony_camera.csv')
                    df = pd.read_csv('./preprocessing/data/kw_data/initial_KW_sony_camera.csv')
                elif str(self.config['CAMPAIGN']['PRODUCT_NAME']) == 'ソニー銀行 住宅ローン':
                    KW_loader = CSVLoader('./preprocessing/data/kw_data/initial_KW_sony_bank_morgage.csv')
                    df = pd.read_csv('./preprocessing/data/kw_data/initial_KW_sony_bank_morgage.csv')
                elif str(self.config['CAMPAIGN']['PRODUCT_NAME']) == 'ソニー Prediction One':
                    KW_loader = CSVLoader('./preprocessing/data/kw_data/initial_KW_sony_po.csv')
                    df = pd.read_csv('./preprocessing/data/kw_data/initial_KW_sony_po.csv')

                else:
                    raise ValueError("Failed to read the PRODUCT_NAME: " + str(self.config['CAMPAIGN']['PRODUCT_NAME']))


                KW_retriever = custermized_trend_retriever(KW_loader, str(self.config['KEY']['OPENAI_EMBEDDING_API_KEY']),  str(self.config['KEY']['OPENAI_EMBEDDING_AZURE_OPENAI_ENDPOINT']))

            
                # 2. define a retriever_tool
                KW_retriever_tool = create_retriever_tool(
                    KW_retriever,
                    str(self.config['TOOL']['GOOD_KW_RETRIEVAL_NAME']),
                    #'Search',
                    str(self.config['TOOL']['GOOD_KW_RETRIEVAL_DISCRPTION']),
                )


                # 4. exampler tool
                exampler_loader = TextLoader(str(self.config['FILE']['EXAMPLER_FILE']))
                exampler_retriever = custermized_trend_retriever(exampler_loader, str(self.config['KEY']['OPENAI_EMBEDDING_API_KEY']),  str(self.config['KEY']['OPENAI_EMBEDDING_AZURE_OPENAI_ENDPOINT'])) 

                # define a retriever_tool
                exampler_retriever_tool = create_retriever_tool(
                    exampler_retriever,
                    str(self.config['TOOL']['RULE_RETRIEVAL_NAME']),
                    #'Search',
                    str(self.config['TOOL']['RULE_RETRIEVAL_DISCRPTION']),
                )
                
                search_tool = load_tools(["serpapi"])
                #search = SerpAPIWrapper()
                # ロードしたツールの中から一番目のものの名前を変更
                # https://book.st-hakky.com/data-science/agents-of-langchain/
                search_tool[0].name = "google_search"
                
                # 3. Initilize LLM and the agent chain
                llm = AzureChatOpenAI(deployment_name="gpt4-0613", openai_api_version="2023-05-15", openai_api_key = str(self.config['KEY']['OPENAI_GPT4_API_KEY']), azure_endpoint = str(self.config['KEY']['OPENAI_GPT4_AZURE_OPENAI_ENDPOINT']), temperature = float(self.config['LLM']['TEMPERATURE']))
                prompt = hub.pull("hwchase17/react")
                
                if int(self.config['LLM']['REACT_VER']) == 1:
                    agent_chain = initialize_agent(    
                        [KW_retriever_tool, search_tool[0],exampler_retriever_tool],
                        llm,
                        agent = AgentType.REACT_DOCSTORE,
                        verbose=True,
                        return_intermediate_steps=True
                    )     

                elif int(self.config['LLM']['REACT_VER']) == 2:
                    tools = [KW_retriever_tool,search_tool[0],exampler_retriever_tool]
                    agent_chain = create_react_agent(
                        llm,
                        tools,
                        prompt
                    )     
                    agent_executor = AgentExecutor(agent=agent_chain, tools=tools, return_intermediate_steps=True, verbose=True)
                
                #print("Reflection is unabled")
                print("the first step")
                
                #while True:
                    
                # read new rejected_kw_list
                with open('./preprocessing/data/string_list.txt', 'r') as file:
                    rejected_kw_list = [line.strip() for line in file]

                # Define the hyperparameters
                num_keywords_per_category = int(self.config['KEYWORD']['NUM_KEYWORDS_PER_CATEGORY'])
                num_new_categories = int(self.config['KEYWORD']['NUM_NEW_CATEGORIES'])
                num_keywords_per_new_category = int(self.config['KEYWORD']['NUM_KEYWORDS_PER_NEW_CATEGORY'])

                rejected_keywords_string = ", ".join(rejected_kw_list)  # Converts list to string
                good_kw_string = ", ".join(good_kw_list)  # Converts list to string

                # 4. Process the first prompt
                # Define the prompt with placeholders for the hyperparameters
                first_prompt = """
                You are a Japanese keyword setting expert for Google search ads for {5} (you can search it on the internet). You will review specific keyword settings for {5}, 
                including the search keywords, their corresponding conversions, cost per conversion ('Cost/conv.'), and clicks.

                I would like you to determine the final keyword list by:
                1. Using google_search (the tool we prepare for you) to find attributes of {5} for which we are delivering ads.
                2. Finding all categories of the keywords and identifying the current keywords for each category.
                3. Using keyword_rule_example_search (the tool we prepare for you) to find the general good examples and rules for the keyword setting for another product and general rule.
                4. By refering the good example and rules, generating {0} more keywords for each category that you think are suitable, considering the attributes of {5}. Do not generate the following keywords: {3}. the folowwing keywords are already verified as good potential keywords: {4}, you can use them as new keyword if they are not in the current keyword lists.
                5. Also generating {1} more categories with category names, each category having {2} new keywords, that you think are suitable keywords for {5}. Do not generate the following keywords: {3}.
                6. Outputting the newly generated Japanese keywords for both existing and new categories (only newly generated keywords without the exsiting ones) in only one dictionary format (including new category and exsiting as we need to parse data) where the key is the category (you need to give an approperate category name to newly generated category) and the value is a string list.

                Generate Keyword with space in japanese if the keyword includes multiple words, such as "ソニー カメラ レンズ" instead of "ソニーカメラレンズ".
                """
                
                # Format the prompt with the hyperparameters
                first_prompt = first_prompt.format(num_keywords_per_category, num_new_categories, num_keywords_per_new_category, rejected_keywords_string, good_kw_string, str(self.config['CAMPAIGN']['PRODUCT_NAME']))
                # 5. Output the first qustion and Run the agent chain

                
                print("Question: " + first_prompt)
                
                if int(self.config['LLM']['REACT_VER']) == 1:
                    action_int_dic, scratch_pad = run_with_retries (agent_chain, first_prompt, int (self.config['LLM']['MAX_ATTEMPTS']))
                elif int(self.config['LLM']['REACT_VER']) == 2:
                    action_int_dic, scratch_pad = run_with_retries (agent_executor, first_prompt, int (self.config['LLM']['MAX_ATTEMPTS']))
                else:
                    raise ValueError("Failed to read the REACT_VER: " + str(e))
                
                # transfer the dic to list by dumping the key
                #new_words_list = list(action_int_dic.values())

                # Initialize an empty list to hold all values
                new_words_list = []
                # Iterate over the dictionary and extend the list with each value list
                for key in action_int_dic:
                    new_words_list.extend(action_int_dic[key])


                # this should be replaced by the func. of Ascade san 
                #new_words_check = cb_kw_plan (new_words_list)
                new_words_check =[ 60, 70, 80, 90, 100, 100, 100, 100, 100]

                # Regular expression to find "Observation 1"
                observation_pattern = r"Observation 1: (.+?)]\n"

                # Find the Observation 1 content
                match = re.search(observation_pattern, scratch_pad, re.DOTALL)

                if match:
                    observation_1_str = match.group(1) + "]"
                    # Convert string representation of list to an actual list
                    observation_1_list = eval(observation_1_str)
                    # Print or use the extracted list
                    print(observation_1_list)
                else:
                    print("Observation 1 not found.")

                
                # if all the element in new_words_check is over 50, break the loop
                if all(x >= 50 for x in new_words_check):
                    
                    # add the new generated keywords to the /data/initial_KW.csv
                    # 1. covert the dic to dataphrame
                    new_keywords_df = pd.DataFrame(
                        [(k, kw) for k, kws in action_int_dic.items() for kw in kws],
                        columns=['Category', 'Keyword']
                    )
                    # List of existing categories in the original DataFrame
                    existing_categories = df['Category'].unique()
                    # Determine if the category is old or new
                    new_keywords_df['Category Status'] = new_keywords_df['Category'].apply(
                        lambda x: 'deeper' if x in existing_categories else 'wider'
                    )

                    # 2. merge the new_keywords_df with the original df
                    df = pd.concat([df, new_keywords_df], ignore_index=True)
                    # 3. replace Nah in click with 0
                    df['Clicks'] = df['Clicks'].fillna(0)
                    # 4. save the new df to the csv file
                    

                    results = find_most_relevant_keywords(new_words_list, self.df_score, 'キーワード', '推定流入数')

                    updated_df = update_clicks(df, results,'Estimated Traffic')

                    updated_df.to_csv('/home/ubuntu/reflexion/New_LLM_Agent_4_Ad_Keyword_and_Text/preprocessing/data/whole_kw.csv', index=False)

                    # calculate mean score which is mean of the click column with the category status is 'deeper' and 'wider'
                    mean_score += updated_df[updated_df['Category Status'].isin(['deeper', 'wider'])]['Clicks'].mean()
                    mean_jacard_score += updated_df[updated_df['Category Status'].isin(['deeper', 'wider'])]['Jacard'].mean()
                    mean_cosine_score += updated_df[updated_df['Category Status'].isin(['deeper', 'wider'])]['Cosine'].mean()
                    mean_bert_score += updated_df[updated_df['Category Status'].isin(['deeper', 'wider'])]['BERT'].mean()
                    # change from dic to list
                    #action_int_list = list(action_int_dic.values())
                    #return action_int_list
                    # or locate the new words whose search check it less than 50
                else:
                    for i in range (len(new_words_check)):
                        if new_words_check[i] < int (self.config['KEYWORD']['SEARCH_CHECK_THRESHOLD']):
                            # add the low search check new words to the tried_kw_list
                            rejected_kw_list.append(str (new_words_list[i]))
                            print ("The new words whose search check is less than 50 is: " + str (new_words_list[i]))
                            # save the rejected_kw_list to a file
                            with open('./preprocessing/data/string_list.txt', 'w') as file:
                                for item in rejected_kw_list:
                                    file.write("%s\n" % item)
                        else: 
                            # add keywords to the good_kw_list
                            good_kw_list.append(str (new_words_list[i]))
                            
                print("Next Round, the rejected low search keywords are: " + str(rejected_kw_list))
                # response = agent_chain ({"input":  first_prompt})
                
                # return action_int_list
        
            elif step < int(self.config['EXE']['GENERATION_ROUND']) and step > 0:
                print("start Step " + str(step))

                KW_loader = CSVLoader('/home/ubuntu/reflexion/New_LLM_Agent_4_Ad_Keyword_and_Text/preprocessing/data/whole_kw.csv')
                KW_retriever = custermized_trend_retriever(KW_loader, str(self.config['KEY']['OPENAI_EMBEDDING_API_KEY']),  str(self.config['KEY']['OPENAI_EMBEDDING_AZURE_OPENAI_ENDPOINT']))

                df = pd.read_csv('/home/ubuntu/reflexion/New_LLM_Agent_4_Ad_Keyword_and_Text/preprocessing/data/whole_kw.csv')
            
                # 2. define a retriever_tool
                KW_retriever_tool = create_retriever_tool(
                    KW_retriever,
                    str(self.config['TOOL']['GOOD_KW_RETRIEVAL_NAME']),
                    #'Search',
                    str(self.config['TOOL']['GOOD_KW_RETRIEVAL_DISCRPTION']),
                )

                # 3. rule tool 
                rule_loader = TextLoader(str(self.config['FILE']['DOMAIN_KNOWLEDGE_FILE']))

                # 4. exampler tool
                exampler_loader = TextLoader(str(self.config['FILE']['EXAMPLER_FILE']))
                exampler_retriever = custermized_trend_retriever(exampler_loader, str(self.config['KEY']['OPENAI_EMBEDDING_API_KEY']),  str(self.config['KEY']['OPENAI_EMBEDDING_AZURE_OPENAI_ENDPOINT'])) 

                # define a retriever_tool
                exampler_retriever_tool = create_retriever_tool(
                    exampler_retriever,
                    str(self.config['TOOL']['RULE_RETRIEVAL_NAME']),
                    #'Search',
                    str(self.config['TOOL']['RULE_RETRIEVAL_DISCRPTION']),
                )
                
                search_tool = load_tools(["serpapi"])
                #search = SerpAPIWrapper()
                # ロードしたツールの中から一番目のものの名前を変更
                # https://book.st-hakky.com/data-science/agents-of-langchain/
                search_tool[0].name = "google_search"
                
                # 3. Initilize LLM and the agent chain
                llm = AzureChatOpenAI(deployment_name="gpt4-0613", openai_api_version="2023-05-15", openai_api_key = str(self.config['KEY']['OPENAI_GPT4_API_KEY']), azure_endpoint = str(self.config['KEY']['OPENAI_GPT4_AZURE_OPENAI_ENDPOINT']), temperature = float(self.config['LLM']['TEMPERATURE']))
                prompt = hub.pull("hwchase17/react")
                
                if int(self.config['LLM']['REACT_VER']) == 1:
                    agent_chain = initialize_agent(    
                        [KW_retriever_tool, search_tool[0],exampler_retriever_tool],
                        llm,
                        agent = AgentType.REACT_DOCSTORE,
                        verbose=True,
                        return_intermediate_steps=True
                    )     

                elif int(self.config['LLM']['REACT_VER']) == 2:
                    tools = [KW_retriever_tool,search_tool[0],exampler_retriever_tool]
                    agent_chain = create_react_agent(
                        llm,
                        tools,
                        prompt
                    )     
                    agent_executor = AgentExecutor(agent=agent_chain, tools=tools, return_intermediate_steps=True, verbose=True)
                    
                # read new rejected_kw_list
                with open('./preprocessing/data/string_list.txt', 'r') as file:
                    rejected_kw_list = [line.strip() for line in file]
                
                # need to find the click growth of each keyword
                # read the whole_kw.csv
                #df_whole = pd.read_csv('/home/ubuntu/reflexion/New_LLM_Agent_4_Ad_Keyword_and_Text/preprocessing/data/whole_kw.csv')
                
                # Merge the two dataframes on 'Keyword' and 'Category'
                #merged_df = pd.merge(df, df_whole, on=['Keyword', 'Category'], suffixes=('_df1', '_df2'))

                # Calculate the difference in Clicks
                #merged_df['Clicks Difference'] = merged_df['Clicks_df2'] - merged_df['Clicks_df1']

                # Group by 'Category Status' and calculate the mean Clicks Difference
                #category_status_mean_difference = merged_df.groupby('Category Status')['Clicks Difference'].mean().reset_index()

                # Filtering and summing clicks for 'wider' and 'deeper' categories
                wider_click_difference = df[df['Category Status'] == 'wider']['Clicks'].sum()
                deeper_click_difference = df[df['Category Status'] == 'deeper']['Clicks'].sum()

                # Filter the dataframe to get the click difference for 'deeper'
                #deeper_click_difference = category_status_mean_difference[category_status_mean_difference['Category Status'] == 'deeper']['Clicks Difference'].values[0]

                # Filter the dataframe to get the click difference for 'wider'
                #wider_click_difference = category_status_mean_difference[category_status_mean_difference['Category Status'] == 'wider']['Clicks Difference'].values[0]

                print("Clicks Difference for 'deeper':", deeper_click_difference)
                print("Clicks Difference for 'wider':", wider_click_difference)

                # Define the hyperparameters
                # Calculate the total sum from the configuration
                total_original_sum = int(self.config['KEYWORD']['NUM_KEYWORDS_PER_CATEGORY']) + int(self.config['KEYWORD']['NUM_NEW_CATEGORIES'])

                # Calculate the total difference
                total_difference = wider_click_difference + deeper_click_difference

                # Calculate the proportion of each click difference
                wider_proportion = wider_click_difference / total_difference
                deeper_proportion = deeper_click_difference / total_difference

                if (str(self.config['KEYWORD']['GENERATION_DYNAMICS']) == 'True'):
                    # Allocate the total sum with a minimum threshold of 1
                    num_keywords_per_category = max(1, int(total_original_sum * wider_proportion))
                    num_new_categories = max(1, int(total_original_sum * deeper_proportion))
                
                elif (str(self.config['KEYWORD']['GENERATION_DYNAMICS']) == 'False'):
              
                    num_keywords_per_category = int(self.config['KEYWORD']['NUM_KEYWORDS_PER_CATEGORY'])
                    num_new_categories = int(self.config['KEYWORD']['NUM_NEW_CATEGORIES'])
                

                # Adjust for rounding errors and maintain the sum
                current_sum = num_keywords_per_category + num_new_categories
                if current_sum != total_original_sum:
                    difference = total_original_sum - current_sum
                    # Adjust the larger proportion to keep both values above 0
                    if num_keywords_per_category > num_new_categories:
                        num_keywords_per_category += difference
                    else:
                        num_new_categories += difference
                num_keywords_per_new_category = int(self.config['KEYWORD']['NUM_KEYWORDS_PER_NEW_CATEGORY'])

                rejected_keywords_string = ", ".join(rejected_kw_list)  # Converts list to string
                good_kw_string = ", ".join(good_kw_list)  # Converts list to string

                # 4. Process the first prompt
                # Define the prompt with placeholders for the hyperparameters
                first_prompt = """
                You are a Japanese keyword setting expert for Google search ads for {5} (you can search it on the internet). You will review specific keyword settings for {5}, 
                including the search keywords, their corresponding conversions, cost per conversion ('Cost/conv.'), and clicks.

                I would like you to determine the final keyword list by:
                1. Using google_search (the tool we prepare for you) to find attributes of {5} for which we are delivering ads.
                2. Finding all categories of the keywords and identifying the current keywords for each category.
                3. Using keyword_rule_example_search (the tool we prepare for you) to find the general good examples and rules for the keyword setting for another product and general rule.
                4. By refering the good example and rules, generating {0} more keywords for each category that you think are suitable, considering the attributes of {5}. Do not generate the following keywords: {3}. the folowwing keywords are already verified as good potential keywords: {4}, you can use them as new keyword if they are not in the current keyword lists.
                5. Also generating {1} more categories with category names, each category having {2} new keywords, that you think are suitable keywords for {5}. Do not generate the following keywords: {3}.
                6. Outputting the newly generated Japanese keywords for both existing and new categories (only newly generated keywords without the exsiting ones) in only one dictionary format (including new category and exsiting as we need to parse data) where the key is the category (you need to give an approperate category name to newly generated category) and the value is a string list.

                Generate Keyword with space in japanese if the keyword includes multiple words, such as "ソニー カメラ レンズ" instead of "ソニーカメラレンズ".
                """
                
                # Format the prompt with the hyperparameters
                first_prompt = first_prompt.format(num_keywords_per_category, num_new_categories, num_keywords_per_new_category, rejected_keywords_string, good_kw_string, str(self.config['CAMPAIGN']['PRODUCT_NAME']))
                # 5. Output the first qustion and Run the agent chain

                
                print("Question: " + first_prompt)
                
                if int(self.config['LLM']['REACT_VER']) == 1:
                    action_int_dic, _ = run_with_retries (agent_chain, first_prompt, int (self.config['LLM']['MAX_ATTEMPTS']))
                elif int(self.config['LLM']['REACT_VER']) == 2:
                    action_int_dic, _ = run_with_retries (agent_executor, first_prompt, int (self.config['LLM']['MAX_ATTEMPTS']))
                else:
                    raise ValueError("Failed to read the REACT_VER: " + str(e))
                
                # transfer the dic to list by dumping the key
                #new_words_list = list(action_int_dic.values())

                # Initialize an empty list to hold all values
                new_words_list = []
                # Iterate over the dictionary and extend the list with each value list
                for key in action_int_dic:
                    new_words_list.extend(action_int_dic[key])


                # this should be replaced by the func. of Ascade san 
                #new_words_check = cb_kw_plan (new_words_list)
                new_words_check =[ 60, 70, 80, 90, 100, 100, 100, 100, 100]
                
                # if all the element in new_words_check is over 50, break the loop
                if all(x >= 50 for x in new_words_check):
                    
                    # add the new generated keywords to the /data/initial_KW.csv
                    # 1. covert the dic to dataphrame
                    new_keywords_df = pd.DataFrame(
                        [(k, kw) for k, kws in action_int_dic.items() for kw in kws],
                        columns=['Category', 'Keyword']
                    )
                    
                    # List of existing categories in the original DataFrame
                    existing_categories = df['Category'].unique()
                    # Determine if the category is old or new
                    new_keywords_df['Category Status'] = new_keywords_df['Category'].apply(
                        lambda x: 'deeper' if x in existing_categories else 'wider'
                    )

                    # 2. merge the new_keywords_df with the original df
                    df = pd.concat([df, new_keywords_df], ignore_index=True)
                    # 3. replace Nah in click with 0
                    df['Clicks'] = df['Clicks'].fillna(0)
                    # 4. save the new df to the csv file
                    #df.to_csv('/home/ubuntu/reflexion/New_LLM_Agent_4_Ad_Keyword_and_Text/preprocessing/data/whole_kw.csv', index=False)


                    # change from dic to list
                    action_int_list = list(action_int_dic.values())


                    results = find_most_relevant_keywords(new_words_list, self.df_score, 'キーワード', '推定流入数')

                    updated_df = update_clicks(df, results, 'Estimated Traffic')

                    updated_df.to_csv('/home/ubuntu/reflexion/New_LLM_Agent_4_Ad_Keyword_and_Text/preprocessing/data/whole_kw.csv', index=False)

                    mean_score += updated_df[updated_df['Category Status'].isin(['deeper', 'wider'])]['Clicks'].mean()
                    mean_jacard_score += updated_df[updated_df['Category Status'].isin(['deeper', 'wider'])]['Jacard'].mean()
                    mean_cosine_score += updated_df[updated_df['Category Status'].isin(['deeper', 'wider'])]['Cosine'].mean()
                    mean_bert_score += updated_df[updated_df['Category Status'].isin(['deeper', 'wider'])]['BERT'].mean()
                    mean_search_volume += updated_df[updated_df['Category Status'].isin(['deeper', 'wider'])]['Search Volume'].mean()
                    mean_competitor_score += updated_df[updated_df['Category Status'].isin(['deeper', 'wider'])]['Competitor Score'].mean()
                    mean_cpc += updated_df[updated_df['Category Status'].isin(['deeper', 'wider'])]['CPC'].mean()


                    #return action_int_list
                # or locate the new words whose search check it less than 50
                else:
                    for i in range (len(new_words_check)):
                        if new_words_check[i] < int (self.config['KEYWORD']['SEARCH_CHECK_THRESHOLD']):
                            # add the low search check new words to the tried_kw_list
                            rejected_kw_list.append(str (new_words_list[i]))
                            print ("The new words whose search check is less than 50 is: " + str (new_words_list[i]))
                            # save the rejected_kw_list to a file
                            
                            with open('./preprocessing/data/string_list.txt', 'w') as file:
                                for item in rejected_kw_list:
                                    file.write("%s\n" % item)
                        else: 
                            # add keywords to the good_kw_list
                            good_kw_list.append(str (new_words_list[i]))
                            
                print("Next Round, the rejected low search keywords are: " + str(rejected_kw_list))
                # response = agent_chain ({"input":  first_prompt})
                
                # return action_int_list
                
                # return action_int_list
            
            else:
                # return the final action list
                keywords_list = updated_df["Keyword"].tolist()
                return mean_score, mean_jacard_score, mean_cosine_score, mean_bert_score, keywords_list, observation_1_str, mean_search_volume, mean_competitor_score, mean_cpc
            step += 1