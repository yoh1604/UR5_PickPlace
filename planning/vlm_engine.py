import os
import json
import base64
import requests
from langfuse.decorators import observe, langfuse_context

# Standard SDKs (Ensure these are installed in your environment)
try:
    from openai import OpenAI
except ImportError:
    pass

try:
    import google.generativeai as genai
except ImportError:
    pass

from .base_vlm import BaseVLMClient 

class VLMEngine(BaseVLMClient):
    def __init__(self, api_key=None, provider="openai", model_name=None):
        super().__init__(api_key, provider, model_name)

    @observe(as_type="generation")
    def get_strategy(self, img_path, user_query):
        
        local = self.provider in ["local", "ollama"]
        ollama_rules = ""
        if local:
            ollama_rules = """
LOCAL MODEL STRICTNESS:
- You are an automated JSON-only API.
- Do not include conversational text.
- Do not wrap JSON in markdown.
- Start exactly with "{" and end exactly with "}".
"""

        prompt_text = f"""
You are an expert robotic vision planner for a UR5e robot.
Your job is to inspect the image and create a safe JSON action_plan.
Look at the lowest point of the objects in the image. Objects whose bottom edges are lower down are physically closer to the camera and must be cleared first if they overlap target behind them.

Primary user request:
"{user_query}"

MULTI-TARGET HANDLING INSTRUCTIONS:
- The user request may ask to pick up multiple different items (e.g., "pick up A and B").
- If multiple items are requested, your action_plan MUST include sequential steps to pick up ALL requested items.
- Order the picking sequence logically (e.g., pick the object that is in the front or unobstructed first).
- If any requested item is blocked by an obstacle, you must include a step to remove the obstacle first before picking the target.

{ollama_rules}

CRITICAL OUTPUT CONTRACT:
- Output only valid JSON.
- Do not use markdown.
- Do not include comments in JSON.
- The executable command source is action_plan only.
- visual_analysis is context only.
- The validator will not rewrite your action_plan, so your action_plan must be correct.

YOLO-WORLD NAMING RULES FOR RELIABLE DETECTION:
1. Every executable target name must be easy for YOLO-World to detect.
2. Use short visual noun phrases: color + generic object category.
3. Prefer 2 to 4 words only.
4. Use lowercase names.
5. DO NOT USE brand names.
6. Avoid vague names such as "object", "item", "thing", "stuff", "container" alone.
7. Avoid relative phrases such as "left object", "front object", "thing near lemon".
8. Avoid overly specific descriptions that YOLO-World may not understand.
9. Good names:
   - "yellow lemon"
   - "red soda can"
   - "white milk carton"
   - "green bottle"
   - "brown bread"
   - "blue cup"
   - "black bowl"
   - "silver spoon"
10. Bad names:
   - "target object"
   - "obstacle"
   - "front occluder"
   - "red cylindrical container with label"
   - "the thing in front of the lemon"
11. The string in action_plan[i].target MUST exactly match one object string in visual_analysis[j].object, except when the target is hidden.
12. If the user says a generic target like "lemon", but the visible object is yellow, output "yellow lemon".
13. If the user says "can", output a visually grounded name like "red soda can" if that is the visible object.
14. If a target is hidden, still use the best YOLO-friendly target name inferred from the user query, e.g. "yellow lemon", not "hidden target".

SCENE ANALYSIS RULES:
1. List every visible tabletop object in visual_analysis.
2. Include support/background objects only when useful, and mark them as:
   "status": "clear_background",
   "target_role": "background"
3. Table, countertop, counter, floor, wall, cabinet, shelf, surface, and background are never executable targets.
4. Do not put background/support objects in action_plan.
5. Look at the lowest point of the objects in the image. Objects whose bottom edges are lower down are physically closer to the camera and must be cleared first ONLY IF they overlap the target (user_query) behind them.

TARGET IDENTIFICATION RULES:
1. Resolve the user's request into one concrete object name from the scene when possible.
2. If the user query has a typo, correct it using visible objects.
   Example: "cam" can mean "can" if a soda can is visible.
3. If the user query expresses intent, map it to a logical visible object.
   Example: "I am thirsty" -> "red soda can", "water bottle", or "white milk carton".
4. resolved_primary_target must be the YOLO-friendly object name.
5. The final primary target step must use the same string as resolved_primary_target whenever the target is visible.
6. The final object picked by the action_plan must correspond to the user's requested object. Do not substitute another object.

OBSTRUCTION RULES:
1. If the primary target is clear, action_plan MUST contain exactly one step:
   pick the primary target.
2. Do not add obstacle-removal steps unless an object physically blocks even if it is only half robot access to the primary target.
3. Treat the primary target as obstructed if another object covers or overlaps the lower part, side part, or front surface area of the primary target.
4. An obstacle must physically overlap, cover, or block the reachable grasp area/path to the primary target.
5. If the target is obstructed, use a sequence:
   - pick the occluding object
   - remove the occluding object
   - pick or search/pick the primary target
6. If the target is hidden, plan a search/removal sequence using the most likely visible occluder, then include a final step to find/pick the original primary target.
7. target_role:
   - "primary_target" only for the object requested by the user
   - "search_occluder" only for objects moved to reveal/search for the primary target
   - "background" only for non-executable background/support surfaces
8. plan_intent:
   - "pick_target" if directly picking the primary target
   - "search_for_hidden_or_blocked_target" if moving occluders first

MULTI-OCCLUDER / ACCESS PATH RULES:
1. The robot approaches the scene from the front, similar to the camera viewpoint.
2. A target is not reachable if there are objects in the frontal access path between the robot/camera and the primary target.
3. If multiple objects form a blocking chain in front of the primary target, remove them from front to back.
4. Do not stop after removing only the nearest/front object if another object still blocks the primary target's lower, central, side, or graspable body region.
5. For a bottle-shaped target, the lower or central body is usually the graspable region. If this region is blocked by a can, box, fruit, or another object, remove that object before picking the bottle.
6. If a small object is in front of another occluding object and blocks the robot's access path to that occluder or target, remove the small front object first.
7. The correct sequence is:
   * remove the nearest/front occluder in the access path
   * remove the next object that still blocks the primary target
   * pick the primary target only after its graspable region is reachable
8. If the primary target is a water bottle and the scene contains a yellow lemon in front, a red soda can behind/near it, and the red soda can blocks the bottle body, the action_plan should be:
   * pick/remove yellow lemon
   * pick/remove red soda can
   * pick blue water bottle
   
FORWARD-LOOKING CAMERA REASONING:
1. The camera observes the scene from the front, not from top-down.
2. In a forward-looking view, an object located lower in the image may be physically in front of another object.
3. If a lower/front object overlaps the bottom part of the requested target, it likely blocks the robot's frontal approach or grasp access.
4. The planner must reason about occlusion using image overlap and visible object ordering:
   - front/lower object = possible occluder
   - rear/upper object = possible target behind it
5. If the target is behind another object and the front object blocks the target's lower or central grasp region, remove the front object first.
6. Look at the lowest point of the objects in the image. Objects whose bottom edges are lower down are physically closer to the camera and must be cleared first if they overlap target behind them.
7. The robotic arm approaches from the front. Never plan a grasp on an object if another object stands taller or overlap 3/4 part and directly in front of it, even if the top of the rear object is visible.

STRICT JSON SCHEMA:
{{
  "resolved_primary_target": "yolo_friendly_object_name",
  "target_visibility": "clear / obstructed / not_exist / hidden",
  "plan_intent": "pick_target / search_for_hidden_or_blocked_target",
  "visual_analysis": [
    {{
      "object": "color generic_object",
      "description": "short visual description",
      "status": "target / obstructing / clear / clear_background",
      "target_role": "primary_target / search_occluder / background"
    }}
  ],
  "action_plan": [
    {{
      "step": 1,
      "action": "pick / remove / search",
      "target": "must_exactly_match_visual_analysis_object_unless_hidden",
      "target_role": "primary_target / search_occluder",
      "step_intent": "pick_primary_target / clear_occluder_to_reveal_primary_target / pick_or_search_primary_target",
      "explanation": "brief reason"
    }}
  ]
}}

FINAL SELF-CHECK BEFORE OUTPUT:
- Is every action_plan target YOLO-friendly?
- Does every visible action_plan target exactly match a visual_analysis object?
- If the target is clear and its graspable lower/front body is accessible, is action_plan exactly one step?
- If the lower/front/graspable part of the primary target is blocked by object A, did you remove object A before picking the target?
- Did you avoid using table/counter/background as an executable target?
- Is the JSON valid?
"""

        # Log to Langfuse
        langfuse_context.update_current_observation(
            name="vlm_scene_understanding",
            model=self.model_name,
            input=prompt_text,
            metadata={
                "provider": self.provider,
                "image_file": str(img_path)
            }
        )
        

        try:
            if self.provider == "openai":
                response_data, input_tokens, output_tokens = self._call_openai(img_path, prompt_text)
            elif self.provider == "gemini":
                response_data, input_tokens, output_tokens = self._call_gemini(img_path, prompt_text)
            elif self.provider == "local" or self.provider == "ollama":
                response_data, input_tokens, output_tokens = self._call_local_vlm(img_path, prompt_text)
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
            return {"error": str(e)}

    def _call_openai(self, img_path, prompt_text):
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

    def _call_gemini(self, img_path, prompt_text):
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

    def _call_local_vlm(self, img_path, prompt_text):
        # 1. Persiapan data
        base64_image = self._encode_image(img_path)
        url = f"{self.ollama_host}/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            # "Accept": "application/json"
        }
        
        # 2. Payload (Ingat: Hapus response_format jika Ollama masih Error 500)
        # Payload yang disesuaikan dengan standar OpenAI-compatible yang didukung Ollama
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
        
        response = requests.post(url, json=payload, headers=headers, timeout=180)
        
        # Debugging: Jika status 500, cetak pesan dari server
        if response.status_code != 200:
            print(f"Error {response.status_code} dari Ollama: {response.text}")
            
        response.raise_for_status()
        data = response.json()
        
        # 4. Parsing response
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        raw_content = message.get("content", "")
        
        try:
            parsed_json = json.loads(raw_content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Gagal memparsing JSON. Output mentah: {raw_content}") from e
        
        usage = data.get("usage", {})
        return parsed_json, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
    