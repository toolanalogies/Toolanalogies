import openai
import sqlite3
import time
import numpy as np
import requests
import backoff
from ..state import State
import anthropic

DB_FILE = "/home/aditya/projects/llm_vpi/api_call.db"
# DB_FILE = "/home/aditya/projects/llm_vpi/label_call.db"

# @backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_time=60, max_tries=5)
def openai_call_func(*args, **kwargs):
    response = openai.chat.completions.create(*args, **kwargs)
    return response

def make_api_call(mode='openai', *args, **kwargs):
    start_time = time.time()
    try:
        # response = openai.ChatCompletion.create(*args, **kwargs)
        if mode == 'openai':
            kwargs.pop('client', None)
            seed = kwargs.pop("seed", None)
            kwargs['seed'] = np.random.randint(0, 100000)
            kwargs['messages'] = kwargs['messages'][1:]
            response = openai_call_func(*args, **kwargs)
        else:
            client = kwargs.pop('client', None)
            kwargs.pop("seed", None)
            kwargs['messages'] = kwargs['messages'][1:]
            kwargs['max_tokens'] = 1024
            response = client.messages.create(*args, **kwargs)
    except Exception as e:
        print("Error in making API call: ", e)
        raise e

    if response:
        # log the response statistics in the database
        try:
            # con = sqlite3.connect(DB_FILE)

            # store the api call information: call_model, time_of_call, prompt_tokens, completion_tokens, total_tokens.
            model_type = response.model
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            if mode == 'openai':
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                content = response.choices[0].message.content
                response_valid_stop = response.choices[0].finish_reason == "stop"
                role = response.choices[0].message.role
            else:
                prompt_tokens = response.usage.input_tokens
                completion_tokens = response.usage.output_tokens
                content = response.content[0].text
                response_valid_stop = response.stop_reason == "end_turn"
                role = response.role

            total_tokens = prompt_tokens + completion_tokens
            # cur = con.cursor()
            # cur.execute("CREATE TABLE IF NOT EXISTS api_calls (call_model TEXT, time_of_call TEXT, prompt_tokens INTEGER, completion_tokens INTEGER, total_tokens INTEGER)")
            # current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            # cur.execute(f"INSERT INTO api_calls VALUES ('{model_type}', '{current_time}', {prompt_tokens}, {completion_tokens}, {total_tokens})")
            # con.commit()
            # con.close()
            # State.n_api_calls += 1
            # State.n_prompt_tokens += prompt_tokens
            # State.n_completion_tokens += completion_tokens
            # State.api_time += time.time() - start_time
        except Exception as e:
            print("Error in logging API call: ", e)
    print("=====================================")
    print("API call made.")
    print(content)
    print("=====================================")
    return content, role, response_valid_stop

    

def parallel_make_api_call(mode='openai', *args, **kwargs):
    try:
        # response = openai.ChatCompletion.create(*args, **kwargs)
        if mode == 'openai':
            kwargs.pop('client', None)
            kwargs['messages'] = kwargs['messages'][1:]
            response = openai_call_func(*args, **kwargs)
        else:
            client = kwargs.pop('client', None)
            kwargs.pop("seed", None)
            kwargs['messages'] = kwargs['messages'][1:]
            kwargs['max_tokens'] = 1024
            response = client.messages.create(*args, **kwargs)
    except Exception as e:
        print("Error in making API call: ", e)
        raise e
    if response:

        model_type = response.model
        # store the api call information: call_model, time_of_call, prompt_tokens, completion_tokens, total_tokens.
        if mode == 'openai':
            content = response.choices[0].message.content
            response_valid_stop = response.choices[0].finish_reason == "stop"
            role = response.choices[0].message.role
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
        else:
            content = response.content[0].text
            response_valid_stop = response.stop_reason == "end_turn"
            role = response.role
            prompt_tokens = response.usage.input_tokens
            completion_tokens = response.usage.output_tokens

        # try:
        #     con = sqlite3.connect(DB_FILE)
        #     total_tokens = prompt_tokens + completion_tokens
        #     cur = con.cursor()
        #     cur.execute("CREATE TABLE IF NOT EXISTS api_calls (call_model TEXT, time_of_call TEXT, prompt_tokens INTEGER, completion_tokens INTEGER, total_tokens INTEGER)")
        #     current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        #     cur.execute(f"INSERT INTO api_calls VALUES ('{model_type}', '{current_time}', {prompt_tokens}, {completion_tokens}, {total_tokens})")
        #     con.commit()
        #     con.close()
        # except Exception as e:
        #     print("Error in logging API call: ", e)

    print("=====================================")
    print("API call made.")
    print(content)
    print("=====================================")
    return content, role, response_valid_stop


def log_all(responses, mode, total_time):
    con = sqlite3.connect(DB_FILE)
    for response in responses:
        # store the api call information: call_model, time_of_call, prompt_tokens, completion_tokens, total_tokens.
        model_type = response.model
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        if mode == 'openai':
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
        else:
            prompt_tokens = response.usage.input_tokens
            completion_tokens = response.usage.output_tokens

        total_tokens = prompt_tokens + completion_tokens
        cur = con.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS api_calls (call_model TEXT, time_of_call TEXT, prompt_tokens INTEGER, completion_tokens INTEGER, total_tokens INTEGER)")
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        cur.execute(f"INSERT INTO api_calls VALUES ('{model_type}', '{current_time}', {prompt_tokens}, {completion_tokens}, {total_tokens})")
        con.commit()
        con.close()
        State.n_api_calls += 1
        State.n_prompt_tokens += prompt_tokens
        State.n_completion_tokens += completion_tokens
        State.api_time += total_time
    print("Logged all API calls.")

