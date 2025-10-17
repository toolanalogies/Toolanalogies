class State:
    n_api_calls = 0
    n_prompt_tokens = 0
    n_completion_tokens = 0
    api_time = 0
    error_count = 0

    def reset():
        State.n_api_calls = 0
        State.n_prompt_tokens = 0
        State.n_completion_tokens = 0
        State.api_time = 0
        State.error_count = 0