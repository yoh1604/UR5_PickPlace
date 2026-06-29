import google.generativeai as genai
import os

# Configure with your API key
genai.configure(api_key="AQ.Ab8RN6LaOWB9GktIMjGGWjBy18wFy9x81UyBohQso22JkKL6VQ") # Or paste your key directly here

print("Available Models:")
for m in genai.list_models():
    # We only care about models that support generation
    if 'generateContent' in m.supported_generation_methods:
        print(f"- {m.name}")