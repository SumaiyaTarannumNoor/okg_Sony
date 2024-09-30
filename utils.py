import pandas as pd
import datetime
import json

def concatenate_llm_parts(config, section, setting_day, observation_period):
    """
    Concatenate all values in the specified section whose keys start with 'PART'.
    parameters:
    - config: the configuration file
    - section: the section in the configuration file
    returns:
    - concatenated_string: the concatenated string of all values in the section whose keys start with 'PART'
    """

    # Retrieve all items (key-value pairs) in the section
    items = config.items(section)
    # Sort items by key to ensure correct order
    sorted_items = sorted(items, key=lambda x: x[0])
    # replace the date named '3/1 and 3/2' with the corresponding value
    setting_day = pd.to_datetime (setting_day)
    observation_period = int(observation_period)
    date_str = ''
    for i in range(1, observation_period + 1):
        # generate the date string
        date_str += str((setting_day - pd.DateOffset(days=i)).strftime('%m/%d')) + ', '
    # delete the last ', '
    date_str = date_str[:-2]
    
    if str(config['EXE']['CPA_MODEL']) == 'False':
        # Concatenate values whose keys start with 'PART'
        if str(config['KEYWORD']['TYPE']) == 'MULTI':
            original_string = "".join(value for key, value in sorted_items if key.startswith('part')and key != 'part2')
        elif str(config['KEYWORD']['TYPE']) == 'SINGLE':
            original_string = "".join(value for key, value in sorted_items if key.startswith('part')and key != 'part3')
        else:
            raise ValueError("The value of 'TYPE' in the 'KEYWORD' section must be either 'MULTI' or 'SINGLE'.")
    else:
        original_string = " ".join(value for key, value in sorted_items if key.startswith('cpa_part'))
    
    
    # replace the date named '3/1 and 3/2' win concatenated_string
    concatenated_string = original_string.replace('3/1 and 3/2', date_str)

    ## replace the date named '3/3' with target date
    concatenated_string = concatenated_string.replace('3/3', str( pd.to_datetime (setting_day).strftime('%m/%d')))

    concatenated_string = concatenated_string.replace('Neural Network Console of Sony', str(config['CAMPAIGN']['PRODUCT_NAME']))

    return concatenated_string, original_string

def concatenate_reflection_beginning(config, section, setting_day, observation_period, setting_times):
    """
    Concatenate all values in the specified section whose keys start with 'PART'.
    parameters:
    - config: the configuration file
    - section: the section in the configuration file
    returns:
    - concatenated_string: the concatenated string of all values in the section whose keys start with 'PART'
    """

    # Retrieve all items (key-value pairs) in the section
    items = config.items(section)
    # Sort items by key to ensure correct order
    sorted_items = sorted(items, key=lambda x: x[0])
    # replace the date named '3/1 and 3/2' with the corresponding value
    setting_day = pd.to_datetime (setting_day)
    observation_period = int(observation_period)
    date_str = ''
    for i in range(1, observation_period + 1):
        # generate the date string
        date_str += str((setting_day - pd.DateOffset(days=i)).strftime('%m/%d')) + ', '
    # delete the last ', '
    date_str = date_str[:-2]
    
    # Concatenate values whose keys start with 'PART'
    original_string = "".join(value for key, value in sorted_items if key.startswith('part'))
    
    # replace the date named '3/1 and 3/2' win concatenated_string
    concatenated_string = original_string.replace('3/1 and 3/2', date_str)

    ## replace the date named '3/3' with target date
    concatenated_string = original_string.replace('3/3', str( pd.to_datetime (setting_day).strftime('%m/%d')))

    return concatenated_string

def parse_output(output):
    """
    Parse the output from the agent chain into a list of integers.
    Parameters:
    - output: the output from the agent chain

    Returns:
    - action_int_list: the list of integers parsed from the output
    """
    # Attempt to parse the output into a list of integers
    return list(map(int, output.split(', ')))


def parse_dic_output(output):
    """
    Parse the output from the agent chain into a list of integers.
    Parameters:
    - output: the output from the agent chain

    Returns:
    - action_int_list: the list of integers parsed from the output
    """
    # Attempt to parse the output into a list of integers
    # Replace single quotes with double quotes to make it valid JSON
    output = output.replace('\'', '\"')
    # delete /n if exists
    output = output.replace('\n', '')
    # Convert the string to a dictionary
    output = json.loads(output)
    
    if type(output) != dict:
        raise ValueError("The output is not a dictionary.")
    return output

def run_with_retries(agent_chain, input_prompt, max_attempts=10):
    """
    Run the agent chain with retries until the output can be successfully parsed.
    Parameters:
    - agent_chain: the agent chain function
    - input_prompt: the input prompt for the agent chain
    - max_attempts: the maximum number of attempts to run the agent chain
    Returns:
    - action_int_list: the list of integers parsed from the output
    """
    
    attempts = 0
    scratch_pad = ''
    while attempts < max_attempts:
        try:
            # Call the function and attempt to parse the output
            response = agent_chain.invoke({"input" : input_prompt})
            #action_int_list = parse_output(response["output"])            
            action_int_list = parse_dic_output(response["output"])
            
            
            # Generate and print the scratchpad content
            scratch_pad = extract_scratchpad_as_string(response)

            return action_int_list, scratch_pad  # Return on successful parsing
        except ValueError as e:
            # Handle the specific parsing error
            print(f"Attempt {attempts + 1} failed: {e}. Retrying...")
            attempts += 1
            # Optional: add a delay here if needed
            import time
            time.sleep(1)  # Sleep for 1 second between retries
    raise Exception("Maximum retries reached, all attempts failed.")

# Function to extract thoughts, actions, observations and return them as a single string
def extract_scratchpad_as_string(response):
    scratchpad_string = f"Input: {response['input']}\n"

    steps = response['intermediate_steps']
    for i, step in enumerate(steps):
        action, observation = step
        scratchpad_string += f"Thought {i+1}: {action.log}\n"
        scratchpad_string += f"Action {i+1}: Tool - {action.tool}, Tool Input - {action.tool_input}\n"
        scratchpad_string += f"Observation {i+1}: {observation}\n\n"

    scratchpad_string += f"Output: {response['output']}"
    
    
    return scratchpad_string

# Function to extract thoughts, actions, observations and return them as a single string with fewer line breaks
def extract_scratchpad_as_compact_string(response):
    scratchpad_string = f"Input: {response['input']}\n"

    steps = response['intermediate_steps']
    for i, step in enumerate(steps):
        action, observation = step
        scratchpad_string += f"\n• Thought {i+1}: {action.log}"
        scratchpad_string += f"\n• Action {i+1}: Tool - {action.tool}, Tool Input - {action.tool_input}"
        scratchpad_string += f"\n• Observation {i+1}: {observation}"

    scratchpad_string += f"\n\nOutput: {response['output']}"
    
    return scratchpad_string


def calculate_ad_settings(setting_daym, observation_period):
    

    # Calculate the day of the month when the ad was first set
    today_day = setting_daym.day
    
    # Calculate how many days have passed since the ad was first set
    days_passed = today_day - 1
    
    # Calculate the number of times the ad has been set
    count = days_passed // observation_period

    
    return max (0, count -1)


def compare_ad_metrics(config, df_current_day, df_day_after):
    
    if str(config['EXE']['CPA_MODEL']) == 'False':
        # Define the columns to compare
        columns_to_compare = [
            'ad1_cost', 'ad1_clicks', 'ad1_Real_CPC',
            'ad2_cost', 'ad2_clicks', 'ad2_Real_CPC',
            'ad3_cost', 'ad3_clicks', 'ad3_Real_CPC',
            'total_clicks', 'total_cost', 'total_CPC'
        ]
    else:
        # Define the columns to compare
        columns_to_compare = [
            'ad1_cost', 'ad1_conversions', 'ad1_Real_CPA',
            'ad2_cost', 'ad2_conversions', 'ad2_Real_CPA',
            'ad3_cost', 'ad3_conversions', 'ad3_Real_CPA',
            'total_conversions', 'total_cost', 'total_CPA'
        ]
    
    # Initialize an empty list to store the description of changes
    changes_description = []

    # Loop through each column and compare the values
    for column in columns_to_compare:
        # Get the values from both dataframes
        value_day_1 = df_current_day[column].iloc[0]
        value_day_2 = df_day_after[column].iloc[0]
        
        # Determine the change description
        if value_day_2 > value_day_1:
            change = 'increased'
        elif value_day_2 < value_day_1:
            change = 'decreased'
        else:
            change = 'remained the same'

        # Append the description to the list
        changes_description.append(f"{column} {change} from {value_day_1} to {value_day_2}")

    # Join all descriptions into a single string
    return '. '.join(changes_description) + '.'

    

def compare_ad_settings(config, df_current_day, df_day_after):
    
    if str(config['EXE']['CPA_MODEL']) == 'False':
        # Define the columns to compare related to budget and CPC settings
        columns_to_compare = [
            'ad1_budget', 'ad1_Max_CPC',
            'ad2_budget', 'ad2_Max_CPC',
            'ad3_budget', 'ad3_Max_CPC'
        ]
    else:
        # Define the columns to compare related to budget and CPC settings
        columns_to_compare = [
            'ad1_budget', 'ad1_target_CPA',
            'ad2_budget', 'ad2_target_CPA',
            'ad3_budget', 'ad3_target_CPA'
        ]

    # Initialize an empty list to store the changes as integers
    changes = []

    # Loop through each column to compare the values
    for column in columns_to_compare:
        # Get the values from both dataframes
        value_day_1 = df_current_day[column].iloc[0]
        value_day_2 = df_day_after[column].iloc[0]
        
        # Determine the change and append the corresponding integer to the list
        if value_day_2 > value_day_1:
            changes.append(1)  # Indicates an increase
        elif value_day_2 < value_day_1:
            changes.append(-1)  # Indicates a decrease
        else:
            changes.append(0)  # Indicates no change

    return changes