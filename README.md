# Ollama Web UI with LiteLLM and Gradio

This project provides a simple web interface to chat with various language models, including local Ollama models, using LiteLLM for model interaction and Gradio for the UI.

## Features

- Select from available Ollama models or other models supported by LiteLLM.
- Interactive chat interface.
- Easy to set up and run.

## Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (for environment management, optional but recommended)
- An Ollama installation (if you want to use local Ollama models). Ensure Ollama is running.

## Setup

1.  **Clone the repository (if applicable) or download the files.**

2.  **Create a virtual environment and install dependencies:**

    It's recommended to use a virtual environment. You can use `uv` or Python's built-in `venv`.

    **Using `uv` (recommended):**
    ```bash
    # Create the virtual environment (e.g., named .venv)
    uv venv --python 3.12
    # Activate the virtual environment
    # On macOS and Linux:
    source .venv/bin/activate
    # On Windows:
    # .venv\Scripts\activate
    
    # Install dependencies
    uv pip install -r requirements.txt
    ```

    **Using `venv`:**
    ```bash
    # Create the virtual environment
    python3 -m venv .venv
    # Activate the virtual environment
    # On macOS and Linux:
    source .venv/bin/activate
    # On Windows:
    # .venv\Scripts\activate
    
    # Install dependencies
    pip install -r requirements.txt
    ```

3.  **Ensure Ollama is running (if using Ollama models):**
    If you plan to use local models with Ollama, make sure your Ollama application is running and you have pulled the models you want to use (e.g., `ollama pull mistral`).

## Running the Application

1.  **Activate your virtual environment** (if you haven't already):
    ```bash
    # If using uv or venv and named .venv
    source .venv/bin/activate 
    ```

2.  **Run the Gradio application:**
    ```bash
    python app.py
    ```

3.  Open your web browser and navigate to the URL provided by Gradio (usually `http://127.0.0.1:7860`).

## How to Use

-   **Select Model:** Choose your desired model from the dropdown menu. The list attempts to populate with your local Ollama models first. If none are found, or if Ollama is not accessible, it will show a default list including `ollama/mistral`, `gpt-3.5-turbo`, and `claude-2`.
-   **Chat:** Type your message in the input box and press Enter or click the submit button (if one were added, currently it's Enter key submission).
-   **Clear Chat:** Click the "Clear Chat" button to reset the conversation.

## Notes

-   For models other than local Ollama models (e.g., `gpt-3.5-turbo`, `claude-2`, models from Groq like `groq/llama3-8b-8192`), you will need to have the respective API keys set up as environment variables for LiteLLM to use them. For example:
    -   `OPENAI_API_KEY` for OpenAI models.
    -   `ANTHROPIC_API_KEY` for Anthropic models.
    -   `GROQ_API_KEY` for Groq models.
    Refer to the [LiteLLM documentation](https://docs.litellm.ai/docs/providers) for more details on provider-specific requirements.
-   The application fetches Ollama models on startup. If you add new Ollama models while the app is running, you might need to restart the Python script to see them in the dropdown. 