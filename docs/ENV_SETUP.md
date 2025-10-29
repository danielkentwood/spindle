# Environment Setup

To use Spindle, you need to set up your API key.

## Create .env file

Create a file named `.env` in the root directory with the following content:

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

If you plan to use OpenAI models in the future, you can also add:

```
OPENAI_API_KEY=your_openai_api_key_here
```

## Security Note

**Never commit your .env file to version control!**

The `.env` file should be added to `.gitignore`.

