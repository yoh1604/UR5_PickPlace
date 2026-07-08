import os
import json
import base64
import requests
from langfuse.decorators import observe, langfuse_context
import io                   # TAMBAHKAN INI
from PIL import Image

try:
    from openai import OpenAI
except ImportError:
    pass

try:
    import google.generativeai as genai
except ImportError:
    pass

from .base_vlm import BaseVLMClient
# Tambahkan import httpx
import httpx

# ... di dalam _validate_local ...


# class LogicValidator():
class LogicValidator(BaseVLMClient):
    def __init__(self, api_key=None, provider="openai", model_name=None):
        super().__init__(api_key, provider, model_name)

    @observe(as_type="generation")
    def validate_strategy(self, image_path, user_query, planner_json):
        
        # Extract the original plan to show the validator explicitly
        original_plan = planner_json.get("action_plan", [])
        
        prompt_text = f"""
You are a Robotic Safety Validator for a UR5e robot manipulating objects on a table.

Your role is ONLY to validate the Planner's proposed action_plan.

You are a gatekeeper.

You MUST NOT modify the Planner output.

# IMPORTANT CONTRACT

You MUST NOT:

* rewrite the action_plan
* add new steps
* remove existing steps
* reorder steps
* replace objects
* create an improved plan
* suggest a new plan

You MUST ONLY decide whether the existing action_plan is acceptable.

Primary User Request

{user_query}

Planner JSON

{json.dumps(planner_json, indent=2)}

Planner action_plan

{json.dumps(original_plan, indent=2)}

IMPORTANT CONTEXT

The robot has already executed all previous steps successfully.

The action_plan below is NOT the original plan.

It is ONLY the remaining steps that have not been executed yet.

DO NOT require previously completed targets to appear again.

Your task is ONLY to judge whether the remaining steps are still logically valid based on the CURRENT scene.

Never fail simply because previous targets are missing from the remaining plan.

Completed steps must be considered already successful.

IMPORTANT FOR MULTI-TARGET REQUESTS

The Planner may decompose a multi-target request into multiple sequential action plans.

The planner may already have completed one or more targets in previous execution steps.

The validator MUST treat the provided action_plan as the CURRENT REMAINING PLAN ONLY.

Do NOT compare the remaining plan against the original user request.

Do NOT require that all original requested targets appear in the remaining plan.

If the remaining plan contains only the unfinished targets, this is correct.

Never fail simply because previously completed targets are absent from the remaining plan.

# Validation Rules

1. User target consistency

The final object picked by the action_plan must correspond to the user's requested object.

Do not silently substitute another object.

Examples:

User:
"I want Coca Cola"

Planner:
pick Sprite

→ FAIL

User:
"I want a Sprite"

Planner:
pick green soda can

Image clearly shows a Sprite can

→ PASS

2. Semantic matching

Minor naming differences are allowed if they refer to the same physical object.

Examples

"sprite"

"green sprite can"

"green soda can"

"soda can"

may refer to the same object.

Do NOT allow semantic matching between different brands or products.

Examples

Coca Cola ≠ Sprite

Water bottle ≠ Soda can

Orange ≠ Lemon

3. Occluders

Objects with

target_role = "search_occluder"

are temporary objects.

They are not the user's desired object.

Removing occluders is acceptable if those objects physically block access to the primary target. or if it is unsafe to directly pick the target since the robot's gripper crash with other.

IF the user request two objects, and one of the target's status is remove then PASS the plan, just check are the request targets are in the action plan 

If the original user request contains multiple targets (e.g. "pick coke and sprite"), assume the planner is allowed to execute them sequentially.

The validator must evaluate ONLY the remaining action_plan that is provided.

Previously completed targets must be considered already finished and must not be required to appear again.

A remaining plan containing only one unfinished target is valid and should PASS if it is safe and logically executable.

4. Safety

Fail the plan if

• a grasped object visually consists of multiple touching objects

• the grasp would simultaneously pick two independent objects

• the selected target is clearly different from the requested object

• an object mentioned in the plan does not exist in the current scene, but if there is a chance to search it then PASS the plan

• the sequence is logically impossible

5. Ignore geometric details

Do NOT evaluate

bbox

pixel coordinates

grasp_pixel_2d

3D coordinates

grasp points

robot trajectories

These values are only hints.

Localization and grasp generation are handled later.

6. Background objects

Never consider these objects removable targets

table

countertop

floor

wall

cabinet

shelf

background

surface

Decision Policy

Return PASS only when

* the final picked object matches the user's request

* all intermediate obstacle removals are reasonable

* the sequence is executable

* the action plan is empty because user's request do not have possibility to be visible 

Return FAIL if

* the primary target is incorrect

* the plan contains hallucinated objects

* the sequence is unsafe

* the sequence is impossible

* the sequence have unnecessary move

Return ONLY valid JSON

{{
"validation_status":"PASS" or "FAIL",
"feedback":"Detailed explanation",
"final_action_plan":[
    exact original action_plan
]
}}
"""
        
        langfuse_context.update_current_observation(
            name="vlm_logic_validation",
            model=self.model_name,
            input=prompt_text,
            metadata={
                "provider": self.provider,
                "image_file": str(image_path)
            }
        )


        try:
            if self.provider == "openai":
                response_data, input_tokens, output_tokens = self._validate_openai(image_path, prompt_text)
            elif self.provider == "gemini":
                response_data, input_tokens, output_tokens = self._validate_gemini(image_path, prompt_text)
            elif self.provider == "local" or self.provider == "ollama":
                response_data, input_tokens, output_tokens = self._validate_local(image_path, prompt_text)
            else:
                raise ValueError(f"Unknown provider: {self.provider}")

            langfuse_context.update_current_observation(
                usage={
                    "input": input_tokens,
                    "output": output_tokens
                },
                output=response_data
            )
            
            return response_data

        except Exception as e:
            langfuse_context.update_current_observation(level="ERROR", status_message=str(e))
            return {"error": str(e), "validation_status": "FAIL"}

    def _validate_openai(self, img_path, prompt_text):
        base64_image = self._encode_image(img_path)
        
        response = self.openai_client.chat.completions.create(
            model=self.model_name,
            response_format={ "type": "json_object" },
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }
            ]
        )
        
        raw_content = response.choices[0].message.content
        parsed_json = json.loads(raw_content)
        
        return parsed_json, response.usage.prompt_tokens, response.usage.completion_tokens

    def _validate_gemini(self, img_path, prompt_text):
        import PIL.Image
        img = PIL.Image.open(img_path)
        
        response = self.gemini_model.generate_content(
            [prompt_text, img],
            generation_config=genai.types.GenerationConfig(response_mime_type="application/json")
        )
        
        parsed_json = json.loads(response.text)
        
        input_tokens = response.usage_metadata.prompt_token_count
        output_tokens = response.usage_metadata.candidates_token_count
        
        return parsed_json, input_tokens, output_tokens

    def _encode_and_resize_image(self, img_path):
        """Mengecilkan gambar sebelum dikirim ke VLM agar VRAM tidak jebol."""
        img = Image.open(img_path)
        img.thumbnail((800, 800)) # Batas aman resolusi
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=80)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def _validate_local(self, img_path, prompt_text):
        # 1. Gunakan fungsi resize yang baru
        base64_image = self._encode_and_resize_image(img_path)
        
        url = f"{self.ollama_host}/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        # 2. Gunakan format payload yang SUDAH TERBUKTI berhasil di test_ollama
        payload = {
            "model": self.model_name,
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=500)
        
        if response.status_code != 200:
            raise RuntimeError(f"Error {response.status_code} dari Ollama: {response.text}")
            
        data = response.json()
        
        # 4. Parsing response
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        raw_content = message.get("content", "")
        
        # Trik Ekstra: LLM lokal sering membungkus JSON dengan ```json ... ```
        # Kita bersihkan teksnya sebelum di parse
        cleaned_content = raw_content.strip()
        if cleaned_content.startswith("```json"):
            cleaned_content = cleaned_content[7:]
        if cleaned_content.startswith("```"):
            cleaned_content = cleaned_content[3:]
        if cleaned_content.endswith("```"):
            cleaned_content = cleaned_content[:-3]
            
        cleaned_content = cleaned_content.strip()
        
        try:
            parsed_json = json.loads(cleaned_content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Gagal memparsing JSON dari VLM lokal.\nOutput mentah:\n{raw_content}") from e
        
        # 5. Ambil metrik token
        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        
        return parsed_json, input_tokens, output_tokens
    
class PostCheckValidator(LogicValidator):


    def validate_target_presence(
            self,
            image_path,
            target
    ):


        prompt=f"""

You are a visual object verifier.


Target:
{target}
Determine if target still exists.

Return ONLY JSON


{{
"target_found_after_action":true,
"confidence":0.91,
"bbox":[120,130,220,310],
"reason":"object visible"
}}

OR

{{
"target_found_after_action":false,
"confidence":0.0,
"bbox":null,
"reason":"object absent"
}}


"""

        if self.provider=="openai":
            result,_,_=self._validate_openai(
                    image_path,
                    prompt
            )

        elif self.provider=="gemini":

            result,_,_=self._validate_gemini(
                    image_path,
                    prompt
            )
        else:

            result,_,_=self._validate_local(
                    image_path,
                    prompt
            )
        return result


