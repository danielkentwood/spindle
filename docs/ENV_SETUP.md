# Environment Setup

To use Spindle, you need to set up your API key.

## Create .env file

Create a file named `.env` in the root directory (it is automatically loaded by `python-dotenv` in the demos and examples) with the following content:

```
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

## Getting an API Key

1. Go to https://console.anthropic.com/
2. Sign up or log in
3. Navigate to API Keys
4. Create a new API key
5. Copy the key and paste it in your `.env` file

## Optional Keys

Add the keys that match the embedding providers you intend to use:

```
# OpenAI embeddings (VectorStore helpers)
OPENAI_API_KEY=your_openai_api_key_here

# Hugging Face Inference API
HF_API_KEY=your_hugging_face_key_here  # alias: HUGGINGFACE_API_KEY

# Gemini embeddings
GEMINI_API_KEY=your_gemini_key_here
```

## Security Note

**Never commit your .env file to version control!**

The `.env` file should be added to `.gitignore`.

