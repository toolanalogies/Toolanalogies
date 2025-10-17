
from .base_prompter import LLMPrompterV1 as BaseLLMPrompter, response_to_snippet, snippet_to_variable
from edit_sys.shape_system.final_annotation import shape_to_file_hier
from edit_sys.shape_system import SINGLE_PASS_API
from .prompts.v8 import edit_extension_with_vision, procedural_with_vision, motion_with_vision
from string import Template
from .one_shot_prompter import APIOnlyVisionPrompter

N_EDITS = 3

class EditExtensionPrompter(APIOnlyVisionPrompter):

    def api_call_get_output(self, shape, edit_request, en_image, n_tries=0):
        messages = self.create_extend_prompts(shape, edit_request, en_image)
        program, response = self._error_tolerant_call(shape, messages, max_tries=self.max_tries, n_tries=n_tries,
                                                         parsing_func=self.parse_edit_extension, error_template_map=None)
        cur_info_list = []
        cur_info = {
            'step': 'edit_extension',
            'prompt': messages[1]['content'],
            'response': response
        }
        cur_info_list.append(cur_info)
        return program, cur_info_list

    def create_extend_prompts(self, shape, edit_request, en_image):

        shape_specification = shape_to_file_hier(shape)
        substitute_dict = {"shape_class": shape.label,
                           "shape_specification": shape_specification,
                           "edit_request": edit_request,
                           "API": SINGLE_PASS_API}
        
        instructions_filled = Template(edit_extension_with_vision.instructions).substitute(substitute_dict)
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
    
    def parse_edit_extension(self, shape, response):
        """
        Parse the initial instructions from the GPT response.
        """
        snippet = response_to_snippet(response)
        program = snippet_to_variable(snippet,shape, "program")
        return program
    

class ProceduralPrompter(APIOnlyVisionPrompter):

    def api_call_get_output(self, shape, edit_request, en_image, n_tries=0):
        messages = self.create_procedural_prompt(shape, edit_request, en_image)
        program, response = self._error_tolerant_call(shape, messages, max_tries=self.max_tries, n_tries=n_tries,
                                                         parsing_func=self.parse_procedural_edits, error_template_map=None)
        cur_info_list = []
        cur_info = {
            'step': 'single_pass',
            'prompt': messages[1]['content'],
            'response': response
        }
        cur_info_list.append(cur_info)
        return program, cur_info_list

    def create_procedural_prompt(self, shape, edit_request, en_image):

        shape_specification = shape_to_file_hier(shape)
        substitute_dict = {"shape_class": shape.label,
                           "shape_specification": shape_specification,
                           "n_edits": N_EDITS}
        
        instructions_filled = Template(procedural_with_vision.instructions).substitute(substitute_dict)
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
    
    def parse_procedural_edits(self, shape, response):
        """
        Parse the initial instructions from the GPT response.
        """
        snippet = response_to_snippet(response)
        program = snippet_to_variable(snippet,shape, "procedural_edits")
        return program
    

class MotionPrompter(APIOnlyVisionPrompter):

    def api_call_get_output(self, shape, edit_request, en_image, n_tries=0):
        messages = self.create_motion_prompt(shape, edit_request, en_image)
        program, response = self._error_tolerant_call(shape, messages, max_tries=self.max_tries, n_tries=n_tries,
                                                         parsing_func=self.parse_motion_edits, error_template_map=None)
        cur_info_list = []
        cur_info = {
            'step': 'single_pass',
            'prompt': messages[1]['content'],
            'response': response
        }
        cur_info_list.append(cur_info)
        return program, cur_info_list

    def create_motion_prompt(self, shape, edit_request, en_image):

        shape_specification = shape_to_file_hier(shape)
        substitute_dict = {"shape_class": shape.label,
                           "shape_specification": shape_specification,
                           "edit_request": edit_request,
                           "API": SINGLE_PASS_API}
        
        instructions_filled = Template(motion_with_vision.instructions).substitute(substitute_dict)
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
    
    def parse_motion_edits(self, shape, response):
        """
        Parse the initial instructions from the GPT response.
        """
        snippet = response_to_snippet(response)
        program = snippet_to_variable(snippet,shape, "motion_edits")
        return program
    