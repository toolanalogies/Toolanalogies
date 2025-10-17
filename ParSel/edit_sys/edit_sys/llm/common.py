import os
print(os.getcwd())
file_name = "edit_sys/llm/.credentials/open_ai.key"
if os.path.exists(file_name):
    OPENAI_KEY = open("edit_sys/llm/.credentials/open_ai.key", "r").readline().strip()
else:
    raise Exception("Please provide the OpenAI key in the file .credentials/open_ai.key")
# OPENAI_MODEL = "gpt-4-turbo-2024-04-09"
OPENAI_MODEL = "gpt-4o"
# MODEL = "gpt-3.5-turbo"
# or
# MODEL = "gpt-4-0125-preview"
# file_name = ".credentials/claude.key"
# if os.path.exists(file_name):
#     CLAUDE_KEY = open(".credentials/claude.key", "r").readline().strip()
# else:
#     raise Exception("Please provide the Claude key in the file .credentials/claude.key")
# CLAUDE_MODEL = "claude-3-opus-20240229"
MODE = "openai"
if MODE == "openai":
    MODEL = OPENAI_MODEL
    KEY = OPENAI_KEY
elif MODE == "claude":
    MODEL = CLAUDE_MODEL
    KEY = CLAUDE_KEY

TEMPERATURE = 1.0
SEED = 43

prompt_cost = {"gpt-4-1106-preview": 0.01,
               "gpt-4-0125-preview": 0.01, 
               "gpt-4-turbo-2024-04-09": 0.01,
               "gpt-4-turbo-preview": 0.01,
               "gpt-4-turbo": 0.01, 
               "gpt-4": 0.03, 
               "gpt-4o": 0.05, 
               "gpt-3.5-turbo": 0.0015,
               "claude-3-opus-20240229": 0.015
               }
completion_cost = {"gpt-4-1106-preview": 0.03, 
                   "gpt-4-0125-preview": 0.03,
                   "gpt-4-turbo-2024-04-09": 0.03,
                   "gpt-4-turbo-preview": 0.03,
                   "gpt-4-turbo": 0.03, 
                   "gpt-4": 0.06, 
                   "gpt-4o": 0.015, 
                   "gpt-3.5-turbo": 0.002,
                   "claude-3-opus-20240229": 0.075
                   }