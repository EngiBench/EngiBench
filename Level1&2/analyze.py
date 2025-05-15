import pandas as pd
import json
import requests
import time
from typing import Optional
import re
from datetime import datetime
import logging
import random
from openai import OpenAI
from together import Together
from ollama import Client
import random
import time
import logging
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
    model = config['models'].get(model_type)
    
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

def call_comparison_model(prompt: str, retry_attempts: int = 10, retry_delay: int = 30) -> Optional[str]:
    for attempt in range(retry_attempts):
        try:
            # Use a placeholder for the API key
            api_key = "YOUR_GEMINI_API_KEY"
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
            headers = {"Content-Type": "application/json"}
            data = {"contents": [{"parts": [{"text": prompt}]}]}
            
            response = requests.post(url, headers=headers, json=data, timeout=30)
            if response.status_code == 200:
                return response.json()['candidates'][0]['content']['parts'][0]['text']
            elif response.status_code in [429, 503]:
                logger.warning(f"Server error (status code: {response.status_code}). Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay * 2)
                continue
            else:
                logger.error(f"API call failed, status code: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Error occurred: {str(e)}")
            if attempt < retry_attempts - 1:
                time.sleep(retry_delay)
            else:
                return None
    return None

def call_answer_model(prompt: str, retry_attempts: int = 10, retry_delay: int = 10, model_type: str = "qwen") -> tuple[Optional[str], Optional[str]]:
    enhanced_prompt = f"""
    Please solve this engineering problem and provide the solution in JSON format:

    Problem:
    {prompt}

    Please provide your solution in the following JSON format:
    {{
        "solution_process": "Your complete solution process",
        "final_answer": "Your answer"
    }}

    Important Notes:
    1. For complex calculations:
       - Keep intermediate steps in fraction form when possible
       - If decimal is necessary, round to 4 decimal places
    2. Show all steps clearly in the solution process
    3. Final answer should be as precise as possible

    Example:
    Problem: "If the airspeed of an airplane is a kilometers per hour and the wind speed is 20 kilometers per hour, what is the difference in kilometers between the distance flown by the airplane against the wind for 3 hours and the distance flown with the wind for 4 hours?"
    
    {{
        "solution_process": "1. With the wind, the effective speed is a + 20 km/h. 2. In 4 hours, the distance flown with the wind is: 4 * (a + 20) = 4a + 80 km. 3. Against the wind, the effective speed is a - 20 km/h. 4. In 3 hours, the distance flown against the wind is: 3 * (a - 20) = 3a - 60 km. 5. The difference in distances is: (4a + 80) - (3a - 60) = 4a + 80 - 3a + 60 = a + 140 km.",
        "final_answer": "a + 140"
    }}

    Ensure your response is in valid JSON format with these exact fields.
    """
    
    for attempt in range(retry_attempts):
        try:
            response = call_api(enhanced_prompt, model_type)
            if response:
                full_response = response['choices'][0]['message']['content']
                try:
                    json_start = full_response.find('{')
                    json_end = full_response.rfind('}') + 1
                    if json_start != -1 and json_end != -1:
                        json_str = full_response[json_start:json_end]
                        json_str = json_str.replace('\\', '\\\\')  # Handle backslashes
                        json_str = json_str.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
                        json_str = re.sub(r'[^\x20-\x7E]', '', json_str)
                        
                        try:
                            response_json = json.loads(json_str)
                        except json.JSONDecodeError as je:
                            logger.error(f"JSON parsing error, cleaned string: {json_str[:100]}...")
                            if attempt < retry_attempts - 1:
                                time.sleep(retry_delay)
                                continue
                            raise je
                            
                        process = response_json.get('solution_process', '').strip()
                        answer = response_json.get('final_answer', '').strip()
                        
                        if process and answer:
                            return process, answer
                except Exception as e:
                    logger.error(f"Error processing response: {str(e)}")
                    if attempt < retry_attempts - 1:
                        time.sleep(retry_delay)
                        continue
            else:
                logger.error("API call returned empty response")
                if attempt < retry_attempts - 1:
                    time.sleep(retry_delay)
                    continue
        except Exception as e:
            logger.error(f"Error occurred: {str(e)}")
            if attempt < retry_attempts - 1:
                time.sleep(retry_delay)
                continue
    return None, None


def compare_answers(generated_answer: str, correct_answer: str) -> Optional[bool]:
    # Maximum retry attempts
    max_retries = 10
    retry_delay = 30  # Retry interval in seconds
    
    for attempt in range(max_retries):
        if not generated_answer or not correct_answer:
            if attempt < max_retries - 1:
                logger.warning(f"Answer is empty, retrying attempt {attempt + 1}...")
                time.sleep(retry_delay)
                continue
            return False
        
        prompt = f"""
        Please analyze these two answers carefully:
        Generated Answer: {generated_answer}
        Standard Answer: {correct_answer}

        Follow these rules for comparison:
        1. For calculation-focused problems:
           - If the numerical values match, consider it correct even if units are missing
           - Focus on the mathematical reasoning and final numerical result
           - Check if the core calculation steps are correct
           - For complex calculations, allow ±2% tolerance in the final numerical result
        
        2. For conceptual or unit-specific problems:
           - Units and their consistency must be considered
           - The complete answer including units is required
        
        3. Consider the answer correct if:
           - The mathematical reasoning is sound
           - The final numerical value matches (within ±2% tolerance for complex calculations)
           - For calculation-focused problems, matching units are not mandatory
        
        Reply only with "True" or "False".
        """
        
        result = call_comparison_model(prompt)
        
        if not result and attempt < max_retries - 1:
            logger.warning(f"API call failed, retrying attempt {attempt + 1}...")
            time.sleep(retry_delay)
            continue
        
        result = result.strip().lower() if result else ""
        print(result)
        if 'false' in result:
            return False
        elif 'true' in result:
            return True
        else:
            if attempt < max_retries - 1:
                logger.warning(f"Result is unclear, retrying attempt {attempt + 1}...")
                time.sleep(retry_delay)
                continue
    
    return None

def save_results(df, generated_model_type, model_type, error_count=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_path = 'evaluation_process'
    
    # Save CSV
    csv_path = f'{base_path}/analyzed_results_{generated_model_type}_{model_type}_{timestamp}.csv'
    df.to_csv(csv_path, index=False)
    logger.info(f"Results saved to CSV: {csv_path}")
    
    # Save XLSX if interrupted due to consecutive errors
    if error_count is not None and error_count >= 9:
        xlsx_path = f'{base_path}/analyzed_results_{generated_model_type}_{model_type}_interrupted_{timestamp}.xlsx'
        df.to_excel(xlsx_path, index=False)
        logger.info(f"Detected {error_count} consecutive errors, results saved to XLSX: {xlsx_path}")

def generate_statistics(df, model_type, count):
    """Generate statistics including accuracy rates"""
    return (
        f"Current progress: {count}/{len(df)} entries ({count/len(df)*100:.1f}%)\n"
        f"Original problem accuracy: {df[f'problem_answer_match_{model_type}'].mean()*100:.2f}%\n"
        f"Converted problem accuracy: {df[f'converted_problem_answer_match_{model_type}'].mean()*100:.2f}%\n"
        f"Knowledge-enhanced problem accuracy: {df[f'enhanced_problem_answer_match_{model_type}'].mean()*100:.2f}%\n"
        f"Rewritten problem accuracy: {df[f'rewritten_problem_answer_match_{model_type}'].mean()*100:.2f}%\n"
        f"Rewritten converted problem accuracy: {df[f'rewritten_converted_problem_answer_match_{model_type}'].mean()*100:.2f}%\n"
        f"Rewritten knowledge-enhanced problem accuracy: {df[f'rewritten_enhanced_problem_answer_match_{model_type}'].mean()*100:.2f}%\n"
    )

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
    eval_prompt = "Please select the model to use (1: GPT-4.1, 2: Claude 3.7 Sonnet, 3: GPT-4.1 Mini, 4: DeepSeek-V3, 5: Gemini 2.5 Flash, 6: Gemini 2.0 Flash, 7: GPT-4.1 Nano, 8: GLM-4-32B, 9: GLM-4-9B, 10: Claude 3.5 Sonnet, 11: Llama 3.3, 12: Qwen2.5-72B, 13: Qwen2.5-7B, 14: Llama 4, 15: DeepSeek-R1 7B, 16: Mixtral-8x7B): "
    model_type = get_model_choice(eval_prompt)
    logger.info(f"Selected model: {model_type}")
    
    gen_prompt = "Please select the model for generation (1: GPT-4.1, 2: GPT-4.1 Mini, 3: DeepSeek-V3, 4: Gemini 2.5 Flash, 5: Gemini 2.0 Flash, 6: GPT-4.1 Nano, 7: GLM-4-32B, 8: GLM-4-9B, 9: Llama 3.3, 10: Qwen2.5-72B, 11: Qwen2.5-7B, 12: Llama 4, 13: DeepSeek-R1 7B, 14: Mixtral-8x7B): "
    generated_model_type = get_model_choice(gen_prompt, is_generation=True)
    logger.info(f"Selected generation model: {generated_model_type}")
    
    # Read CSV file
    logger.info("Starting to read CSV file...")
    df = pd.read_csv('level12_dataset.csv')
    df_filtered = df.copy()
    df_filtered.reset_index(drop=True, inplace=True)

    logger.info(f"Found {len(df_filtered)} entries to process")
    
    # Initialize new columns
    new_columns = [
        f'problem_llm_process_{model_type}', f'problem_llm_answer_{model_type}', f'problem_answer_match_{model_type}',
        f'converted_problem_llm_process_{model_type}', f'converted_problem_llm_answer_{model_type}', f'converted_problem_answer_match_{model_type}',
        f'enhanced_problem_llm_process_{model_type}', f'enhanced_problem_llm_answer_{model_type}', f'enhanced_problem_answer_match_{model_type}',
        f'rewritten_problem_llm_process_{model_type}', f'rewritten_problem_llm_answer_{model_type}', f'rewritten_problem_answer_match_{model_type}',
        f'rewritten_converted_problem_llm_process_{model_type}', f'rewritten_converted_problem_llm_answer_{model_type}', f'rewritten_converted_problem_answer_match_{model_type}',
        f'rewritten_enhanced_problem_llm_process_{model_type}', f'rewritten_enhanced_problem_llm_answer_{model_type}', f'rewritten_enhanced_problem_answer_match_{model_type}'
    ]
    
    for col in new_columns:
        df_filtered[col] = None
    
    total = len(df_filtered)
    consecutive_errors = 0  # Counter for consecutive errors
    processed_count = 0  # Counter for processed entries
    
    for idx, row in df_filtered.iterrows():
        processed_count += 1
        logger.info(f"Processing entry {idx+1}/{total}")
        
        try:
            # Process original problem
            process, answer = generated_model(row['problem'], model_type=model_type)
            if process is None or answer is None:
                raise Exception("Failed to process original problem")
            df_filtered.at[idx, f'problem_llm_process_{model_type}'] = process
            df_filtered.at[idx, f'problem_llm_answer_{model_type}'] = answer
            df_filtered.at[idx, f'problem_answer_match_{model_type}'] = compare_answers(answer, str(row['answer']))

            # Process converted problem
            process, answer = generated_model(row[f'converted_problem'], model_type=model_type)
            if process is None or answer is None:
                raise Exception("Failed to process converted problem")
            df_filtered.at[idx, f'converted_problem_llm_process_{model_type}'] = process
            df_filtered.at[idx, f'converted_problem_llm_answer_{model_type}'] = answer
            df_filtered.at[idx, f'converted_problem_answer_match_{model_type}'] = compare_answers(answer, str(row['converted_problem_llm_answer']))

            # Process knowledge-enhanced problem
            process, answer = generated_model(row[f'knowledge_enhanced_problem'], model_type=model_type)
            if process is None or answer is None:
                raise Exception("Failed to process knowledge-enhanced problem")
            df_filtered.at[idx, f'enhanced_problem_llm_process_{model_type}'] = process
            df_filtered.at[idx, f'enhanced_problem_llm_answer_{model_type}'] = answer
            df_filtered.at[idx, f'enhanced_problem_answer_match_{model_type}'] = compare_answers(answer, str(row['answer']))

            # Process rewritten problem
            process, answer = generated_model(row[f'rewritten_problem'], model_type=model_type)
            if process is None or answer is None:
                raise Exception("Failed to process rewritten problem")
            df_filtered.at[idx, f'rewritten_problem_llm_process_{model_type}'] = process
            df_filtered.at[idx, f'rewritten_problem_llm_answer_{model_type}'] = answer
            df_filtered.at[idx, f'rewritten_problem_answer_match_{model_type}'] = compare_answers(answer, str(row['rewritten_answer']))

            # Process rewritten converted problem
            process, answer = generated_model(row[f'rewritten_converted_problem'], model_type=model_type)
            if process is None or answer is None:
                raise Exception("Failed to process rewritten converted problem")
            df_filtered.at[idx, f'rewritten_converted_problem_llm_process_{model_type}'] = process
            df_filtered.at[idx, f'rewritten_converted_problem_llm_answer_{model_type}'] = answer
            df_filtered.at[idx, f'rewritten_converted_problem_answer_match_{model_type}'] = compare_answers(answer, str(row['rewritten_converted_problem_llm_answer']))

            # Process rewritten knowledge-enhanced problem
            process, answer = generated_model(row[f'rewritten_knowledge_enhanced_problem'], model_type=model_type)
            if process is None or answer is None:
                raise Exception("Failed to process rewritten knowledge-enhanced problem")
            df_filtered.at[idx, f'rewritten_enhanced_problem_llm_process_{model_type}'] = process
            df_filtered.at[idx, f'rewritten_enhanced_problem_llm_answer_{model_type}'] = answer
            df_filtered.at[idx, f'rewritten_enhanced_problem_answer_match_{model_type}'] = compare_answers(answer, str(row['rewritten_answer']))

            consecutive_errors = 0  # Reset consecutive error counter
            
        except Exception as e:
            logger.error(f"Error occurred while processing entry {idx+1}: {str(e)}")
            consecutive_errors += 1
            
            if consecutive_errors >= 9:
                logger.error(f"Detected {consecutive_errors} consecutive errors, interrupting processing and saving results")
                save_results(df_filtered, generated_model_type, model_type, consecutive_errors)
                return

        # Save and output statistics every 20 entries processed
        if processed_count % 20 == 0:
            stats = generate_statistics(df_filtered, model_type, processed_count)
            logger.info(f"\n=== Progress saved [{processed_count}/{total}] ===\n{stats}")

        # Save and output statistics every 50 entries processed
        if processed_count % 50 == 0:
            save_results(df_filtered, generated_model_type, model_type)
            stats = generate_statistics(df_filtered, model_type, processed_count)
            logger.info(f"\n=== Progress saved [{processed_count}/{total}] ===\n{stats}")
        
        time.sleep(5)  # Avoid API rate limits
    
    # Final save
    save_results(df_filtered, generated_model_type, model_type)
    
    # Output statistics
    logger.info("\n=== Statistics ===")
    logger.info(f"Total entries processed: {total}")
    logger.info(f"Original problem accuracy: {df_filtered[f'problem_answer_match_{model_type}'].mean()*100:.2f}%")
    logger.info(f"Converted problem accuracy: {df_filtered[f'converted_problem_answer_match_{model_type}'].mean()*100:.2f}%")
    logger.info(f"Knowledge-enhanced problem accuracy: {df_filtered[f'enhanced_problem_answer_match_{model_type}'].mean()*100:.2f}%")
    logger.info(f"Rewritten problem accuracy: {df_filtered[f'rewritten_problem_answer_match_{model_type}'].mean()*100:.2f}%")
    logger.info(f"Rewritten converted problem accuracy: {df_filtered[f'rewritten_converted_problem_answer_match_{model_type}'].mean()*100:.2f}%")
    logger.info(f"Rewritten knowledge-enhanced problem accuracy: {df_filtered[f'rewritten_enhanced_problem_answer_match_{model_type}'].mean()*100:.2f}%")

if __name__ == "__main__":
    main()