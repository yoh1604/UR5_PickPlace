import os
import json
import base64
import requests
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

# Tambahkan import httpx
import httpx



class BaseVLMClient:
    """Kelas dasar untuk mengurus koneksi API ke berbagai provider AI."""
    
    DEFAULT_MODELS = {
        "openai": "gpt-4o",
        "gemini": "gemini-2.5-pro",
        "ollama": "qwen3.5:27b",
        "local": "qwen3.5:27b"
    }

    def __init__(self, api_key=None, provider="openai", model_name=None):
        self.provider = provider.lower()
        
        # Smart Default: Pilih model yang cocok dengan provider jika tidak diisi
        self.model_name = model_name or self.DEFAULT_MODELS.get(self.provider, "gpt-4o")

        # Otomatis tarik API Key dari .env jika tidak dipassing
        if self.provider == "openai":
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        elif self.provider == "gemini":
            self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        else:
            self.api_key = api_key

        self.ollama_host = os.getenv("OLLAMA_HOST_URL").rstrip("/")

        self._setup_client()

    def _setup_client(self):
        if self.provider == "openai" and self.api_key:
            self.openai_client = OpenAI(api_key=self.api_key)
        elif self.provider == "gemini" and self.api_key:
            genai.configure(api_key=self.api_key)
            self.gemini_model = genai.GenerativeModel(self.model_name)

    def _encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
        
    def _encode_and_resize_image(self, img_path):
        """Mengecilkan gambar sebelum dikirim ke VLM agar VRAM tidak jebol."""
        img = Image.open(img_path)
        img.thumbnail((800, 800)) # Batas aman resolusi
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=80)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def _generate_json_response(self, img_path, prompt_text):
        """Metode universal untuk memanggil provider yang sesuai."""
        if self.provider == "openai":
            return self._call_openai(img_path, prompt_text)
        elif self.provider == "gemini":
            return self._call_gemini(img_path, prompt_text)
        elif self.provider in ["local", "ollama"]:
            return self._call_local_vlm(img_path, prompt_text)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

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
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
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
            # CATATAN PENTING: Jangan gunakan "response_format": {"type": "json_object"} 
            # karena Qwen di Ollama sering crash dengan parameter itu. 
            # Prompt kamu sudah menyuruhnya mengembalikan JSON.
        }
        
        # 3. Kirim request dengan timeout panjang (180 detik)
        response = requests.post(url, json=payload, headers=headers, timeout=180)
        
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

    # def _call_local_vlm(self, img_path, prompt_text):
    #     base64_image = self._encode_image(img_path)
    #     url = f"{self.ollama_host}/v1/chat/completions"
    #     # url = f"{self.ollama_host}/api/chat"
    #     payload = {
    #         "model": self.model_name,
    #         "format": "json",
    #         "stream": False,
    #         "messages": [
    #             {"role": "user", "content": prompt_text, "images": [base64_image]}
    #         ]
    #     }
    #     response = requests.post(url, json=payload)
    #     response.raise_for_status()
    #     data = response.json()
    #     parsed_json = json.loads(data["message"]["content"])
    #     input_tokens = data.get("prompt_eval_count", 0)
    #     output_tokens = data.get("eval_count", 0)
    #     return parsed_json, input_tokens, output_tokens