import io
import json
import ollama
import random
import numpy as np
import time
import datetime
import pytz
import argparse
import pandas as pd 
import datetime
import os
import logging
from typing import Any, Dict, List
from datasets import load_dataset
from bart_score import BARTScorer
bart_scorer = BARTScorer(device='cpu', checkpoint='facebook/bart-large-cnn')
bart_scorer.load(path='bart_score.pth')

# Define a new file path in the home directory
home_directory = os.path.expanduser('~')
current_datetime = datetime.datetime.now()


    
def main(logger, model, num_test, num_predict, temperature, seed):



    logger = logger 

    #load alapca model
    def _make_r_io_base(f, mode: str):
        if not isinstance(f, io.IOBase):
            f = open(f, mode=mode)
        return f

    #load json file 
    def jload(f, mode="r"):
        """Load a .json file into a dictionary."""
        f = _make_r_io_base(f, mode)
        jdict = json.load(f)
        f.close()
        return jdict

    #make a csv file for averaged metrics
    def avg_make_csv(args, avg_token_sec, avg_num_input_tokens, avg_num_output_tokens, avg_inference_time, 
                     avg_bart, total_inference_time, total_num_output_tokens, avg_TTFS):
        num_prompts = args.num_test
        num_predict = args.num_predict
        model = args.model
        avg_TTFS = avg_TTFS
        avg_token_sec = avg_token_sec 
        avg_num_input_tokens = avg_num_input_tokens
        avg_num_output_tokens = avg_num_output_tokens
        avg_inference_time = avg_inference_time 
        avg_bart = avg_bart
        total_inference_time = total_inference_time
        total_num_output_tokens = total_num_output_tokens
        data = {
            'num_prompts' : [num_prompts],
            'avg_TTFS': [avg_TTFS],
            'avg_token_sec' : [avg_token_sec],
            'avg_num_input_tokens' : [avg_num_input_tokens],
            'avg_num_output_tokens' : [avg_num_output_tokens],
            'avg_inference_time' : [avg_inference_time],
            'avg_bart' : [avg_bart],
            'total_inference_time' : [total_inference_time],
            'total_num_output_tokens' : [total_num_output_tokens],
            'Maximum length of response': [num_predict],
            'model_name': [model]
            }
        return data 
    
    #make a csv file for metric of each prompt
    def per_prompt_make_csv(args, prompt_index, inference_time, token_sec,num_input_tokens , num_output_token, TTFS, bart):
        num_prompts = args.num_test
        num_predict = args.num_predict
        model = args.model       
        TTFS = TTFS 
        prompts = prompt_index
        TTFS_list = []
        inference_list = []
        num_output_token_list = []
        num_input_token_list=[]
        token_sec_list = []
        bart_list = []

        for key in prompts:
            TTFS_list.append(TTFS[key])
            inference_list.append(inference_time[key])
            num_input_token_list.append(num_input_tokens[key])
            num_output_token_list.append(num_output_token[key])
            token_sec_list.append(token_sec[key])
            bart_list.append(bart[key])
        data = {
            'Prompt': prompts,
            'TTFS': TTFS_list,
            'Inference latency': inference_list,
            'Num_input_tokens' : num_input_token_list,
            'Num_output_tokens': num_output_token_list,
            'Token/sec': token_sec_list,
            'BART' : bart_list
            # 'Maximum length of response': [num_predict],
            # 'model_name': [model]
        }

        return data
    
    def write_csv(avg_data, prompt_data, filename, home_directory ):

        file_path = os.path.join(home_directory, filename)

        # Try saving the DataFrame

        avg_df = pd.DataFrame(avg_data)
        prompt_df = pd.DataFrame(prompt_data)

        try:
            avg_df.to_csv(file_path+'_avg', index =False)
            prompt_df.to_csv(file_path+'_prompt', index= False)
        except PermissionError as e:
            print(f"PermissionError: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")

    def time_convert(iso_timestamp):
# Convert the ISO 8601 timestamp to a datetime object in UTC
        dt_utc = datetime.datetime.fromisoformat(iso_timestamp.rstrip('Z')).replace(tzinfo=pytz.UTC)

        # Convert the datetime object to Eastern Time if running on pc
        eastern = pytz.timezone('US/Eastern')
        dt_eastern = dt_utc.astimezone(eastern)

        # Convert the datetime object to a timestamp (seconds since epoch)
        timestamp = dt_eastern.timestamp()

        # Convert the timestamp to time.time() style
        time_style = time.mktime(dt_eastern.timetuple()) + dt_eastern.microsecond / 1e6

        ##Use UTC if running on a phone
        # dt = datetime.datetime.fromisoformat(iso_timestamp.rstrip('Z'))

        # # Convert the datetime object to a timestamp (seconds since epoch)
        # timestamp = dt.timestamp()

        # # Convert the timestamp to time.time() style
        # time_style = time.mktime(dt.timetuple()) + dt.microsecond / 1e6

        return time_style
    
    def rouge_fn(targets: List[List[str]], responses: List[str]) -> Dict[str, float]:
        """Computes ROUGE by taking the max ROUGE-N per reference and N."""
        # Following strategy from https://www.aclweb.org/anthology/W04-1013/.
        # Identify best reference per response and ROUGE type.
        rouge_types = ["rouge1", "rouge2", "rougeLsum"]
        max_references = {rouge_type: [] for rouge_type in rouge_types}
        for targ_for_resp, resp in zip(targets, responses):
            # Compute individual scores per example/ref pair.
            resp_scores = [metrics.rouge([t], [resp]) for t in targ_for_resp]
            # Find best scoring references for generated output and ROUGE type.
            for rouge_type in rouge_types:
                best_score_index = max(
                    range(len(resp_scores)), key=lambda x: resp_scores[x][rouge_type]
                )
                best_ref = targ_for_resp[best_score_index]
                # Add the reference to the new reference list.
                max_references[rouge_type].append(best_ref)
        # Compute metric for each of the reference lists for a ref type.
        results = {}
        for rouge_type in rouge_types:
            results[rouge_type] = metrics.rouge(max_references[rouge_type], responses)[
                rouge_type
            ]
        return results

    #This is the mixinstruct dataset
    ds = load_dataset("llm-blender/mix-instruct")
    # data = ds['validation']
    data = ds['train']
    output_response = {}
    TTFS = {}
    num_output_tokens = {}
    num_input_tokens ={}
    token_sec = {}
    inference_time = {}
    bart = {}

    current_datetime = datetime.datetime.now()
    filename = str(current_datetime) +'_'+ str(args.model)+'_'+ str(args.num_test)+'_' + str(num_predict)+'_' + str(temperature)+ '_'+ str(seed)

    #these are indices for input prompts
    candidate_index = range(0, num_test) 
    # test_index = np.random.choice(candidate_index, num_test, replace= False)
    test_index = candidate_index

    generated_tokens = 0
    elapsed_time = 0
    processed_prompts = 0
    prompt_set = {}
    for idx in test_index:
        prompt_set[idx] =  data['instruction'][idx] + data['input'][idx]

    for idx in test_index:
        #Used ollama.generate to measure the metrics, it returns json file with token/sec, inference time, # of output tokens ...etc
        stream = ollama.generate(model = model,  prompt =  data['instruction'][idx] + data['input'][idx], stream = True, options={"temperature" : temperature, "seed": seed,"num_predict" : num_predict})
        idx = int(idx)

        now = time.time()
        pred =''
        for foo, chunk in enumerate(stream):
            pred += chunk['response']
            if foo == 0:
                first_token_timestamp = time_convert(chunk['created_at'])
                TTFS[idx] = first_token_timestamp - now
            elif chunk['done'] == True:
                # last_token_timestamp = time_convert(chunk['created_at'])
                # print(f"Input Prompt: {input_prompts[idx]}")
                # print(f"Output: {output}")
                label = data['output'][idx]
                # print(f"pred: {pred}, label: {label}")

                bart_score = bart_scorer.score([prompt_set[idx]], [label], batch_size=4)[0]

                token_sec[idx] =  chunk['eval_count']/chunk['total_duration']*(10**9)
                num_input_tokens[idx] =  chunk['prompt_eval_count']
                num_output_tokens[idx] = chunk['eval_count']
                inference_time[idx] = chunk['total_duration']/(10**9)
                bart[idx] = bart_score

                generated_tokens += num_output_tokens[idx]
                elapsed_time += inference_time[idx]
                processed_prompts += 1
                print(f"Processed_prompts: {processed_prompts}, BART: {bart_score}, Generated tokens: {generated_tokens}, Elapsed time: {elapsed_time}, TTFS: {TTFS[idx]}")
                logger.info(f"Gnerated tokens: {generated_tokens}, Elapsed time: {elapsed_time}, Num_processed_prompts: {processed_prompts}")

    ##calculate the average metrics
    avg_TTFS = 0
    avg_token_sec = 0
    avg_num_input_tokens = 0
    avg_num_output_tokens = 0
    avg_inference_time = 0
    total_inference_time = 0
    total_num_output_tokens = 0
    avg_bart = 0

    for idx in test_index:
        avg_TTFS += TTFS[idx]/num_test
        avg_token_sec += token_sec[idx]/num_test
        avg_num_input_tokens += num_input_tokens[idx]/num_test
        avg_num_output_tokens += num_output_tokens[idx]/num_test
        avg_inference_time += inference_time[idx]/num_test
        avg_bart += bart[idx]/num_test
        total_inference_time += inference_time[idx]
        total_num_output_tokens += num_output_tokens[idx]

    print(f"avg_TTFS: {avg_TTFS}")
    print('avg_token_sec: ', avg_token_sec)
    print('avg_num_input_tokens: ', avg_num_input_tokens)
    print('avg_num_output_tokens: ', avg_num_output_tokens)
    print('avg_inference_time: ', avg_inference_time)
    print(f"avg bart {avg_bart}")
    print('total_inference_time: ', total_inference_time)
    print('total_output_tokens: ', total_num_output_tokens)
    print('all inference time', inference_time)
    print('all number of output tokens', num_output_tokens)

    avg_df = avg_make_csv(args, avg_token_sec, avg_num_input_tokens, avg_num_output_tokens, avg_inference_time, 
                          avg_bart, total_inference_time, total_num_output_tokens, avg_TTFS)
    prompt_df = per_prompt_make_csv(args, test_index, inference_time, token_sec, num_input_tokens ,num_output_tokens, TTFS, bart)
    write_csv(avg_df, prompt_df, filename, home_directory)

if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser(description='Process Inputs.')

    # Add arguments
    parser.add_argument('--model', type=str, help='model name', default='qwen2:7b')
    parser.add_argument('--num_test', type=int, help='number of samples for measurement', default = 100000)
    parser.add_argument('--num_predict', type = int, help = 'how many tokens should be generated', default = 450)
    parser.add_argument('--temperature', type = float, help = 'creativity of the model', default = 0.7)
    parser.add_argument('--seed', type = int, default = 0)
    # Parse the arguments
    args = parser.parse_args()
    
    #Set file path for a logger
    file_path = os.path.join(home_directory, str(args.model)+str(current_datetime))

    #reproductivility
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Configure logging
    logging.basicConfig(level=logging.INFO,
                        handlers=[logging.FileHandler(file_path, 'w', 'utf-8')])
    logging.getLogger("httpx").setLevel(logging.WARNING)
    # Create a logger object
    logger = logging.getLogger(__name__)


    main(logger, args.model, args.num_test, args.num_predict, args.temperature, args.seed)