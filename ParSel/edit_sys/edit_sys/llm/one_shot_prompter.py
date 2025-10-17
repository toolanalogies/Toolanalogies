"""
This are a few of our baseline methods. (no analytical edit propagation)
"""
from .base_prompter import LLMPrompterV1 as BaseLLMPrompter, response_to_snippet, snippet_to_variable
from edit_sys.shape_system.final_annotation import shape_to_file_hier
from edit_sys.shape_system import SINGLE_PASS_API
from .prompts.v8 import single_call_prompt, single_call_with_vision
from string import Template


class APIOnlyLLMPrompter(BaseLLMPrompter):
    # Generate in single pass
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_tries = 10 # just let it try more often

    def api_call_single_pass_program(self, shape, edit_request, n_tries=0):
        
        messages = self.create_single_pass_prompt(shape, edit_request)
        program, response = self._error_tolerant_call(shape, messages, max_tries=self.max_tries, n_tries=n_tries,
                                                         parsing_func=self.parse_single_pass_program, error_template_map=None)
        cur_info_list = []
        cur_info = {
            'step': 'single_pass',
            'prompt': messages[1]['content'],
            'response': response
        }
        cur_info_list.append(cur_info)
        return program, cur_info_list

    def create_single_pass_prompt(self, shape, edit_request):
        shape_specification = shape_to_file_hier(shape)
        substitute_dict = {"shape_class": shape.label,
                           "shape_specification": shape_specification,
                           "edit_request": edit_request,
                           "API": SINGLE_PASS_API}
        
        instructions_filled = Template(single_call_prompt.instructions).substitute(substitute_dict)
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": instructions_filled}
        ]
        
        return messages
    
    def parse_single_pass_program(self, shape, response):
        """
        Parse the initial instructions from the GPT response.
        """
        snippet = response_to_snippet(response)
        program = snippet_to_variable(snippet,shape, "program")
        return program
    
class APIOnlyVisionPrompter(APIOnlyLLMPrompter):

    def api_call_single_pass_program(self, shape, edit_request, en_image, n_tries=0):
        messages = self.create_single_pass_prompt(shape, edit_request, en_image)
        program, response = self._error_tolerant_call(shape, messages, max_tries=self.max_tries, n_tries=n_tries,
                                                         parsing_func=self.parse_single_pass_program, error_template_map=None)
        cur_info_list = []
        cur_info = {
            'step': 'single_pass',
            'prompt': messages[1]['content'],
            'response': response
        }
        cur_info_list.append(cur_info)
        return program, cur_info_list

    def create_single_pass_prompt(self, shape, edit_request, en_image):

        shape_specification = shape_to_file_hier(shape)
        substitute_dict = {"shape_class": shape.label,
                           "shape_specification": shape_specification,
                           "edit_request": edit_request,
                           "API": SINGLE_PASS_API}
        
        instructions_filled = Template(single_call_with_vision.instructions).substitute(substitute_dict)
        messages = [

            {"role": "system", "content": ""},
            {"role": "user", 
             "content": [
                 {
                     "type": "text",
                     "text": instructions_filled
                 },
                 {
                     "type": "image_url",
                     "image_url":{
                         "url": f"data:image/jpeg;base64,{en_image}"
                     }
                 }
             ]}
        ]

        return messages