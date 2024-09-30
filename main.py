import configparser
import os
import re

from metrics import evaluate_keywords_against_paragraph

from transformers import logging
logging.set_verbosity_error()


from okg_agent import okg_agent


if __name__ == "__main__":
    
    ## 0. Specify the configuration file
    config_file_path = './config.ini'
    config_0 = configparser.ConfigParser()
    config_0.read('./config.ini')
   
    ## 1. Initialize the agent
    agent = okg_agent(config_file_path)

    ## 2. Run the agent
    mean_score, mean_jacard_score, mean_cosine_score, mean_bert_score, keyword_list, observation_1_str, mean_search_volume, mean_competitor_score, mean_cpc = agent.run()
    
    ## 3. Evaluate the keywords
    bleu2, bleu4, rouge1, bertScore = evaluate_keywords_against_paragraph(observation_1_str, keyword_list)
    
    print("The final keywords are: " + str(keyword_list))
    print("The final traffic score is: " + str(mean_score))
    print("The final jacard score is: " + str(mean_jacard_score/int(config_0['EXE']['GENERATION_ROUND'])))
    print("The final cosine score is: " + str(mean_cosine_score/int(config_0['EXE']['GENERATION_ROUND'])))
    print("The final bert score is: " + str(mean_bert_score/int(config_0['EXE']['GENERATION_ROUND'])))
    print("The final bleu4 score is: " + str(bleu4))
    print("The final bleu2 score is: " + str(bleu2))
    print("The final rouge1 score is: " + str(rouge1))
    print("The final bert score is: " + str(bertScore))
    print ("The final search volume is: " + str(mean_search_volume))
    print ("The final competitor score is: " + str(mean_competitor_score))
    print ("The final cpc is: " + str(mean_cpc))

    ## 4(optional): Save the results to a csv file
    #result_df = pd.DataFrame({
        #'Keyword': keyword_list,
        #'Traffic Score': mean_score,
        #'Jacard Score': mean_jacard_score/int(config_0['EXE']['GENERATION_ROUND']),
        #'Cosine Score': mean_cosine_score/int(config_0['EXE']['GENERATION_ROUND']),
        #'BERT Score': mean_bert_score/int(config_0['EXE']['GENERATION_ROUND']),
        #'BLEU-2 Score': bleu2,
        #'BLEU-4 Score': bleu4,
        #'ROUGE-1 Score': rouge1,
        #'BERT Score': bertScore,
        #'Search Volume': mean_search_volume,
        #'Competitor Score': mean_competitor_score,
        #'CPC': mean_cpc
    #})
    # Save the DataFrame to a CSV file, the name of the csv file is the same as the product name
    #result_df.to_csv(f'./results/{config_0["CAMPAIGN"]["PRODUCT_NAME"]}_results.csv', index=False)
    
    
  