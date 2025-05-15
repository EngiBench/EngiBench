import pandas as pd
import json
import requests
import time
import random
from typing import Optional
import re
from datetime import datetime
import logging
from openai import OpenAI
from ollama import Client
from google import genai
import anthropic
from model_imports import (
    call_gpt41_api, call_claude37_sonnet_api, call_gpt41_mini_api,
    call_deepseek_v3_api, call_gemini25_flash_api, call_gemini20_flash_api,
    call_gpt41_nano_api, call_glm4_32b_api, call_glm4_9b_api,
    call_claude35_sonnet_api, call_llama33_api, call_qwen25_72b_api,
    call_qwen25_7b_api, call_llama4_api, call_deepseek_r1_7b_api,
    call_mixtral_8x7b_api
)

# Load configuration
with open('config.json') as config_file:
    config = json.load(config_file)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def call_api(prompt, model_type):
    api_key = config['api_keys'].get(model_type)
    model_type = config['models'].get(model_type)
    
    if model_type.lower() == "gpt-4.1":
        return call_gpt41_api(prompt, api_key)
    elif model_type.lower() == "claude-3.7-sonnet":
        return call_claude37_sonnet_api(prompt, api_key)
    elif model_type.lower() == "gpt-4.1-mini":
        return call_gpt41_mini_api(prompt, api_key)
    elif model_type.lower() == "deepseek-v3":
        return call_deepseek_v3_api(prompt, api_key)
    elif model_type.lower() == "gemini-2.5-flash":
        return call_gemini25_flash_api(prompt, api_key)
    elif model_type.lower() == "gemini-2.0-flash":
        return call_gemini20_flash_api(prompt, api_key)
    elif model_type.lower() == "gpt-4.1-nano":
        return call_gpt41_nano_api(prompt, api_key)
    elif model_type.lower() == "glm-4-32b":
        return call_glm4_32b_api(prompt, api_key)
    elif model_type.lower() == "glm-4-9b":
        return call_glm4_9b_api(prompt, api_key)
    elif model_type.lower() == "claude-3.5-sonnet":
        return call_claude35_sonnet_api(prompt, api_key)
    elif model_type.lower() == "llama-3.3":
        return call_llama33_api(prompt, api_key)
    elif model_type.lower() == "qwen-2.5-72b":
        return call_qwen25_72b_api(prompt, api_key)
    elif model_type.lower() == "qwen-2.5-7b":
        return call_qwen25_7b_api(prompt, api_key)
    elif model_type.lower() == "llama-4":
        return call_llama4_api(prompt, api_key)
    elif model_type.lower() == "deepseek-r1-7b":
        return call_deepseek_r1_7b_api(prompt, api_key)
    elif model_type.lower() == "mixtral-8x7b":
        return call_mixtral_8x7b_api(prompt, api_key)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def call_answer_model(prompt: str, model_type: str = "qwen", retry_attempts: int = 10, retry_delay: int = 15) -> Optional[str]:
    enhanced_prompt = f"""
    Please solve this modeling problem:

    Problem:
    {prompt}

    Important Notes:
    1. Provide a clear and structured solution
    2. Consider all relevant factors and constraints
    3. Show your modeling process and reasoning
    """
    
    for attempt in range(retry_attempts):
        try:
            response = call_api(enhanced_prompt, model_type)
            if response:
                return response
                
            if attempt < retry_attempts - 1:
                time.sleep(retry_delay)
                continue
        except Exception as e:
            logger.error(f"Error occurred (Attempt {attempt+1}/{retry_attempts}): {str(e)}")
            if attempt < retry_attempts - 1:
                time.sleep(retry_delay)
                continue
    
    logger.error(f"All {retry_attempts} attempts failed. Returning None.")
    return None

def call_scoring_model(answer: str, score_type: str, score_criteria: str, model_type: str = "qwen", retry_attempts: int = 5, retry_delay: int = 5) -> Optional[dict]:
    if not answer:
        print(f"Warning: Empty answer for {score_type}")
        return None
        
    prompt = f"""
    You are a professional modeling competition judge with extensive experience in evaluating mathematical and engineering models. Please conduct a rigorous evaluation of the following answer based on the provided criteria.

    Answer to evaluate:
    {answer}

    Evaluation Criteria:
    {score_criteria}

    Please evaluate strictly according to the criteria and provide your assessment in the following JSON format:
    {{
        "score": <score between 0-10, can use decimal points for precision>,
        "reason": "Detailed evaluation breakdown:\n
                  1. [Specific criterion] - [sub-score] points: [justification]\n
                  2. [Specific criterion] - [sub-score] points: [justification]\n
                  3. [Specific criterion] - [sub-score] points: [justification]\n
                  Final score: [total] points"
    }}

    Note: 
    - Break down your scoring into specific components
    - Provide clear justification for each sub-score
    - Be objective and consistent in your evaluation
    - Consider both the technical accuracy and the methodology
    """

    for attempt in range(retry_attempts):
        try:
            response = call_api(prompt, model_type)
            if response:
                full_response = response                
                # Clean special characters from response text
                cleaned_response = ''.join(char for char in full_response if ord(char) >= 32 or char in ['\n', '\r', '\t'])
                
                try:
                    # Try to parse the entire response
                    result = json.loads(cleaned_response)
                    # Verify the result format is correct
                    if 'score' not in result or 'reason' not in result:
                        logger.warning(f"Warning: Missing required fields in JSON response for {score_type} (Attempt {attempt+1}/{retry_attempts})")
                        logger.warning(f"Received: {result}")
                        if attempt < retry_attempts - 1:
                            time.sleep(retry_delay)
                            continue
                    return result
                except json.JSONDecodeError as json_err:
                    logger.error(f"JSON parsing error for {score_type} (Attempt {attempt+1}/{retry_attempts}): {str(json_err)}")
                    # If failed, try to extract JSON part
                    json_start = cleaned_response.find('{')
                    json_end = cleaned_response.rfind('}') + 1
                    if json_start != -1 and json_end != -1:
                        json_str = cleaned_response[json_start:json_end]
                        try:
                            # Further clean JSON string
                            json_str = json_str.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
                            json_str = re.sub(r'[^\x20-\x7E]', '', json_str)
                            # Try to fix common JSON format issues
                            json_str = json_str.replace('\"', '"').replace('""', '"')
                            result = json.loads(json_str)
                            # Verify the result format is correct
                            if 'score' not in result or 'reason' not in result:
                                logger.warning(f"Warning: Missing required fields in extracted JSON for {score_type} (Attempt {attempt+1}/{retry_attempts})")
                                if attempt < retry_attempts - 1:
                                    time.sleep(retry_delay)
                                    continue
                            return result
                        except json.JSONDecodeError as extract_err:
                            logger.error(f"Failed to extract valid JSON for {score_type} (Attempt {attempt+1}/{retry_attempts}): {str(extract_err)}")
                            logger.error(f"Extracted JSON string: {json_str[:100]}...")
                            if attempt < retry_attempts - 1:
                                time.sleep(retry_delay)
                                continue
            else:
                logger.error(f"API call failed for {score_type} (Attempt {attempt+1}/{retry_attempts})")
                if attempt < retry_attempts - 1:
                    time.sleep(retry_delay)
                    continue
        except Exception as e:
            logger.error(f"Error in scoring {score_type} (Attempt {attempt+1}/{retry_attempts}): {str(e)}")
            if attempt < retry_attempts - 1:
                time.sleep(retry_delay)
                continue
    
    logger.error(f"All {retry_attempts} attempts failed for {score_type}. Returning None.")
    return None
    

def process_level3_dataset(file_path, answer_model_type="qwen", scoring_model_type="qwen", regenerate_answer=True, existing_csv_file=None):
    logger.info(f"Using answer generation model: {answer_model_type}")
    logger.info(f"Using scoring model: {scoring_model_type}")
    df = pd.read_excel(file_path)
    
    results = []
    total_problems = len(df)
    
    for idx, row in df.iterrows():
        if (idx + 1) % 10 == 0:
            print(f"\n{'='*50}")
            print(f"Processing problem {idx + 1}/{total_problems}")
            print(f"{'='*50}\n")
            # Save temporary results every 10 entries
            temp_save_path = f"temp_results_{idx + 1}_{answer_model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            pd.DataFrame(results).to_csv(temp_save_path, index=False)
            logger.info(f"Temporary results saved to {temp_save_path}")
        
        # Combine questions
        original_problem = f"{row['question']} {row['subquestion']}"
        rewritten_problem = f"{row['question_modified']} {row['subquestion_modified']}"
        
        # Get answers
        if regenerate_answer:
            original_answer = call_answer_model(original_problem, model_type=answer_model_type)
            rewritten_answer = call_answer_model(rewritten_problem, model_type=answer_model_type)
        else:
            # Read answers from existing CSV file
            csv_file = existing_csv_file if existing_csv_file else f"level3_results_test20250420_{answer_model_type}.csv"
            logger.info(f"Reading answers from existing file {csv_file}")
            try:
                df_exist = pd.read_csv(csv_file)
                if idx < len(df_exist):
                    original_answer = df_exist["llm_original_answer"].iloc[idx]
                    rewritten_answer = df_exist["llm_rewritten_answer"].iloc[idx]
                    logger.info(f"Successfully read answer for row {idx+1}")
                else:
                    logger.warning(f"No row with index {idx} in CSV file, regenerating answers")
                    original_answer = call_answer_model(original_problem, model_type=answer_model_type)
                    rewritten_answer = call_answer_model(rewritten_problem, model_type=answer_model_type)
            except Exception as e:
                logger.error(f"Error reading CSV file: {str(e)}, regenerating answers")
                original_answer = call_answer_model(original_problem, model_type=answer_model_type)
                rewritten_answer = call_answer_model(rewritten_problem, model_type=answer_model_type)
        
        
        result = {
            'original_problem': original_problem,
            'rewritten_problem': rewritten_problem,
            'llm_original_answer': original_answer,
            'llm_rewritten_answer': rewritten_answer
        }
        
        # Score types and corresponding column names
        score_columns = [
            'redundant_information_filtering_score',
            'multi_objective_tradeoff_score',
            'uncertainty_handling_score',
            'deep_knowledge_integration_score'
        ]
        
        # Evaluate each score type
        for score_column in score_columns:
            if pd.notna(row[score_column]):
                # Extract score type from column name (remove _score suffix)
                score_type = score_column.replace('_score', '')
                score_criteria = row[score_column]  # Get scoring criteria
                
                # Score original answer
                print(f"Processing {score_type} for original answer of problem {idx + 1}")
                original_score_result = call_scoring_model(original_answer, score_type, score_criteria, scoring_model_type)
                
                # Score rewritten answer
                print(f"Processing {score_type} for rewritten answer of problem {idx + 1}")
                rewritten_score_result = call_scoring_model(rewritten_answer, score_type, score_criteria, scoring_model_type)
                
                # Save original answer's score result
                if original_score_result:
                    result[f'{score_type}_original_score'] = original_score_result['score']
                    result[f'{score_type}_original_reason'] = original_score_result['reason']
                    print(f"  ✓ Successfully scored original answer {score_type}: {original_score_result['score']}")
                else:
                    result[f'{score_type}_original_score'] = None
                    result[f'{score_type}_original_reason'] = None
                    print(f"  ✗ Failed to score original answer {score_type}. Setting to None.")
                
                # Save rewritten answer's score result
                if rewritten_score_result:
                    result[f'{score_type}_rewritten_score'] = rewritten_score_result['score']
                    result[f'{score_type}_rewritten_reason'] = rewritten_score_result['reason']
                    print(f"  ✓ Successfully scored rewritten answer {score_type}: {rewritten_score_result['score']}")
                else:
                    result[f'{score_type}_rewritten_score'] = None
                    result[f'{score_type}_rewritten_reason'] = None
                    print(f"  ✗ Failed to score rewritten answer {score_type}. Setting to None.")
                    
                # Record answer length for debugging
                if original_answer:
                    answer_length = len(original_answer)
                    print(f"  Original answer length: {answer_length} characters")
                else:
                    print("  Original answer is None or empty")
                    
                if rewritten_answer:
                    answer_length = len(rewritten_answer)
                    print(f"  Rewritten answer length: {answer_length} characters")
                else:
                    print("  Rewritten answer is None or empty")
        
        results.append(result)
    
    return results

# Model mapping dictionary
MODEL_MAPPING = {
    '1': 'gpt-4.1',
    '2': 'claude-3.7-sonnet',
    '3': 'gpt-4.1-mini',
    '4': 'deepseek-v3',
    '5': 'gemini-2.5-flash',
    '6': 'gemini-2.0-flash',
    '7': 'gpt-4.1-nano',
    '8': 'glm-4-32b',
    '9': 'glm-4-9b',
    '10': 'claude-3.5-sonnet',
    '11': 'llama-3.3',
    '12': 'qwen-2.5-72b',
    '13': 'qwen-2.5-7b',
    '14': 'llama-4',
    '15': 'deepseek-r1-7b',
    '16': 'mixtral-8x7b'
}

def get_model_choice(prompt, is_generation=False):
    """Elegantly get user's model choice"""
    model_type = None
    while True:
        try:
            choice = input(prompt).strip()
            if choice in MODEL_MAPPING:
                model_type = MODEL_MAPPING[choice]
                if is_generation and model_type in ['claude-3.7-sonnet', 'claude-3.5-sonnet']:
                    print("Claude models are not available for generation")
                    continue
                break
            else:
                print(f"Invalid choice, please enter a number between 1 and {len(MODEL_MAPPING)}")
        except Exception as e:
            logger.error(f"Error getting model choice: {str(e)}")
    return model_type

def main():
    # Select answer generation model
    eval_prompt = "Please select the answer generation model (1: GPT-4.1, 2: Claude 3.7 Sonnet, 3: GPT-4.1 Mini, 4: DeepSeek-V3, 5: Gemini 2.5 Flash, 6: Gemini 2.0 Flash, 7: GPT-4.1 Nano, 8: GLM-4-32B, 9: GLM-4-9B, 10: Claude 3.5 Sonnet, 11: Llama 3.3, 12: Qwen2.5-72B, 13: Qwen2.5-7B, 14: Llama 4, 15: DeepSeek-R1 7B, 16: Mixtral-8x7B): "
    answer_model_type = get_model_choice(eval_prompt)
    logger.info(f"Selected answer generation model: {answer_model_type}")
    
    # Select scoring model
    scoring_prompt = "Please select the scoring model (1: GPT-4.1, 2: Claude 3.7 Sonnet, 3: GPT-4.1 Mini, 4: DeepSeek-V3, 5: Gemini 2.5 Flash, 6: Gemini 2.0 Flash, 7: GPT-4.1 Nano, 8: GLM-4-32B, 9: GLM-4-9B, 10: Claude 3.5 Sonnet, 11: Llama 3.3, 12: Qwen2.5-72B, 13: Qwen2.5-7B, 14: Llama 4, 15: DeepSeek-R1 7B, 16: Mixtral-8x7B): "
    scoring_model_type = get_model_choice(scoring_prompt)
    logger.info(f"Selected scoring model: {scoring_model_type}")
    
    # Ask whether to regenerate answers
    existing_csv_file = 'existing_csv_file'
    while True:
        regenerate_choice = input("Regenerate answers? (T: Yes, F: No, read from existing file): ").strip().upper()
        if regenerate_choice in ['T', 'F']:
            regenerate_answer = (regenerate_choice == 'T')
            logger.info(f"Answer regeneration {'enabled' if regenerate_answer else 'disabled'}")
            if not regenerate_answer:
                # If not regenerating, specify CSV file
                default_csv = existing_csv_file
                csv_input = input(f"Enter CSV filename to read (default: {default_csv}): ").strip()
                existing_csv_file = csv_input if csv_input else default_csv
                logger.info(f"Reading existing answers from {existing_csv_file}")
            break
        else:
            print("Invalid choice, please enter T or F")
    
    file_path = 'level3_modeling.xlsx'
    results = process_level3_dataset(file_path, answer_model_type, scoring_model_type, regenerate_answer=regenerate_answer, existing_csv_file=existing_csv_file)
    
    current_date = datetime.now().strftime('%Y%m%d')
    output_path = f'level3_results_test_{current_date}_{scoring_model_type}_{answer_model_type}.csv'
    # Create DataFrame and save
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_path, index=False)
    
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()
