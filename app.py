import gradio as gr
from litellm import completion
import ollama
import logging
import base64
import io
from PIL import Image

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DEFAULT_ALT_TEXT_FOR_IMAGE_ONLY = "(Uploaded Image)" # Constant for default alt text

# --- Provider and Model Definitions ---
def get_ollama_models():
    try:
        models_data = ollama.list().get("models", [])
        model_names = [m["name"] for m in models_data]
        logging.info(f"Successfully fetched Ollama models: {model_names}")
        return model_names
    except Exception as e:
        logging.error(f"Error fetching Ollama models: {e}", exc_info=True)
        return []

# User's specific Ollama models to be used as fallback
USER_OLLAMA_FALLBACK_MODELS = [
    "devstral:24b", "llama3.3:70b", "llama3.2:latest", "qwen3:32b", 
    "qwq:32b", "gemma3:27b", "deepseek-r1:14b", "qwen2.5vl:32b"
]

# Define providers and their models
# The model names here should be what LiteLLM expects
# For Ollama, we'll fetch dynamically and then prefix them in the update logic if needed by LiteLLM
# or assume LiteLLM handles "ollama/" + model_name if provider is specified.

PREDEFINED_PROVIDERS = {
    "ollama": {
        "fetch_func": get_ollama_models,
        "prefix": "ollama/", # LiteLLM generally expects ollama/model_name
        "dynamic_fetch": True,
        "fallback_models": USER_OLLAMA_FALLBACK_MODELS
    },
    "openrouter": {
        "models": [
            "openai/gpt-4o",
            "google/gemini-pro-1.5",
            "mistralai/mistral-large-latest",
            "anthropic/claude-3-opus",
            "nousresearch/nous-hermes-2-mixtral-8x7b-dpo",
            "meta-llama/llama-3-70b-instruct",
            "fireworks/firefunction-v1" # Example with a different structure
        ],
        "prefix": "openrouter/", # LiteLLM expects this full prefix
        "dynamic_fetch": False
    },
    "openai": {
        "models": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "gpt-4o"],
        "prefix": "", # LiteLLM recognizes these directly
        "dynamic_fetch": False
    },
    "groq": {
        "models": ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it"],
        "prefix": "groq/",
        "dynamic_fetch": False
    },
    "anthropic": {
        "models": ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307", "claude-2.1"],
        "prefix": "", # LiteLLM recognizes these directly
        "dynamic_fetch": False
    }
}

PROVIDER_NAMES = list(PREDEFINED_PROVIDERS.keys())

def get_models_for_provider(provider_name):
    if provider_name not in PREDEFINED_PROVIDERS:
        return []
    
    provider_info = PREDEFINED_PROVIDERS[provider_name]
    raw_models = []
    used_fallback = False

    if provider_info.get("dynamic_fetch", False):
        fetched_models = provider_info["fetch_func"]()
        if fetched_models:
            raw_models = fetched_models
        elif "fallback_models" in provider_info:
            logging.warning(f"Dynamic fetch for {provider_name} failed or returned no models. Using predefined fallback models for this provider.")
            raw_models = provider_info["fallback_models"]
            used_fallback = True
        else:
            logging.warning(f"Dynamic fetch for {provider_name} failed and no fallback models defined for this provider.")
            raw_models = []
    else:
        raw_models = provider_info.get("models", [])
    
    prefix = provider_info.get("prefix", "")
    # Ensure the prefix is not doubly applied if a model name itself already contains it (though less likely with this structure)
    models = []
    for m in raw_models:
        if prefix and m.startswith(prefix):
            # This case is if the model name in the list already has the full prefix (e.g. for openrouter if we listed openrouter/openai/gpt-4o)
            # However, our current setup for openrouter has prefix="openrouter/" and models like "openai/gpt-4o"
            # So the standard path below should apply.
             models.append(m)
        elif prefix:
            models.append(f"{prefix}{m}")
        else:
            models.append(m)
            
    if used_fallback:
        logging.info(f"Using fallback models for provider '{provider_name}': {models}")
    else:
        logging.info(f"Models for provider '{provider_name}': {models}")
    return models

# --- Image to Base64 Conversion ---
def image_to_base64(pil_image, format="JPEG"):
    if pil_image is None: return None
    buffered = io.BytesIO()
    pil_image.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# --- Chat Logic ---
def chat_with_model(selected_model_name, user_text, user_image_pil, history_state):
    history_for_chatbot_display = list(history_state)
    messages_for_llm = []

    # Reconstruct messages for LiteLLM from history_state
    for user_turn_display_item, bot_turn_text in history_for_chatbot_display:
        llm_user_content_parts = []
        if isinstance(user_turn_display_item, tuple): # User turn was (PIL.Image, displayed_text_caption)
            img_pil_from_hist, displayed_text_caption = user_turn_display_item
            img_b64_from_hist = image_to_base64(img_pil_from_hist)
            if img_b64_from_hist:
                llm_user_content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64_from_hist}"}})
            if displayed_text_caption != DEFAULT_ALT_TEXT_FOR_IMAGE_ONLY: # Only add text if it wasn't our default placeholder
                llm_user_content_parts.append({"type": "text", "text": displayed_text_caption})
        elif user_turn_display_item: # User turn was just text (string)
            llm_user_content_parts.append({"type": "text", "text": user_turn_display_item})
        
        if llm_user_content_parts:
            messages_for_llm.append({"role": "user", "content": llm_user_content_parts})
        if bot_turn_text: # Bot responses are text
            messages_for_llm.append({"role": "assistant", "content": bot_turn_text})

    # Current user message (for LiteLLM)
    current_llm_user_content_parts = []
    actual_user_text = user_text.strip() if user_text else ""
    if actual_user_text:
        current_llm_user_content_parts.append({"type": "text", "text": actual_user_text})
    user_image_b64 = image_to_base64(user_image_pil)
    if user_image_b64:
        current_llm_user_content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{user_image_b64}"}})
    
    if not current_llm_user_content_parts: # No text and no image for current turn
        if not messages_for_llm: # No history and no current input
             yield history_for_chatbot_display, history_for_chatbot_display # No change, effectively an error caught by submit handler
             return
    else:
        messages_for_llm.append({"role": "user", "content": current_llm_user_content_parts})

    # Prepare user input for display in chatbot history
    user_display_turn_item = None
    if user_image_pil:
        user_display_turn_item = (user_image_pil, actual_user_text if actual_user_text else DEFAULT_ALT_TEXT_FOR_IMAGE_ONLY)
    elif actual_user_text:
        user_display_turn_item = actual_user_text
    # If user_display_turn_item is still None here, it means empty input, handled by submitter.

    if not selected_model_name:
        error_msg = "Error: No model selected or specified."
        if user_display_turn_item is not None: # Add error to history if there was some input
            history_for_chatbot_display.append((user_display_turn_item, error_msg))
        yield history_for_chatbot_display, history_for_chatbot_display
        return

    logging.info(f"Attempting to stream chat with model: {selected_model_name}")
    # logging.debug(f"Messages for LiteLLM: {messages_for_llm}") # Can be very verbose with base64

    if user_display_turn_item is not None:
        history_for_chatbot_display.append((user_display_turn_item, "")) # Placeholder for bot response
    else: # Should not happen if submit handler catches empty inputs
        yield history_for_chatbot_display, history_for_chatbot_display
        return
        
    try:
        response_stream = completion(model=selected_model_name, messages=messages_for_llm, stream=True)
        current_bot_response_text = ""
        for chunk in response_stream:
            delta = chunk.choices[0].delta.content
            if delta:
                current_bot_response_text += delta
                history_for_chatbot_display[-1] = (user_display_turn_item, current_bot_response_text)
                yield history_for_chatbot_display, history_for_chatbot_display
        logging.info(f"Successfully streamed response from {selected_model_name}")
    except Exception as e:
        logging.error(f"Error during model streaming with {selected_model_name}: {e}", exc_info=True)
        error_detail = f"Error interacting with model {selected_model_name}: {str(e)}\n\nCommon issues:\n"
        error_detail += "- Ensure the model name is correct (e.g., 'ollama/mistral', 'openrouter/openai/gpt-4o').\n"
        error_detail += "- For API-based models, ensure API keys (e.g., OPENAI_API_KEY, OPENROUTER_API_KEY) are set as environment variables.\n"
        error_detail += "- Check if the model server (e.g., Ollama for local models) is running and accessible.\n"
        error_detail += "- The model might be overloaded or unavailable, or not support the input type (e.g. vision for text-only model)."
        history_for_chatbot_display[-1] = (user_display_turn_item, error_detail)
        yield history_for_chatbot_display, history_for_chatbot_display

# --- Gradio UI ---    
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Ollama & LiteLLM Chat Interface with Vision")
    
    state_history = gr.State([])
    # Hidden state to store the fully qualified model name from dropdowns or manual input
    selected_model_for_chat = gr.State(None)

    with gr.Row():
        provider_dropdown = gr.Dropdown(
            label="Select Provider", 
            choices=PROVIDER_NAMES, 
            value=PROVIDER_NAMES[0] if PROVIDER_NAMES else None,
            scale=1
        )
        model_dropdown = gr.Dropdown(
            label="Select Model", 
            choices=[], # Will be populated based on provider
            scale=2,
            interactive=True
        )
        manual_model_textbox = gr.Textbox(
            label="Manually Enter Full Model String (e.g., openrouter/google/gemini-pro)",
            placeholder="Overrides dropdowns if filled",
            scale=2
        )
        refresh_ollama_button = gr.Button("🔄 Refresh Ollama Models", scale=1, visible=(PROVIDER_NAMES[0] == "ollama" if PROVIDER_NAMES else False))

    chatbot = gr.Chatbot(label="Chat Conversation", height=450, bubble_full_width=False)
    
    with gr.Row():
        with gr.Column(scale=3):
            msg_textbox = gr.Textbox(
                label="Your Message", 
                placeholder="Type your message or upload an image...", 
                show_label=False, container=False
            )
        with gr.Column(scale=1, min_width=160):
            image_upload = gr.Image(type="pil", label="Upload Image (Optional)", height=100, show_label=False)
            # clear_image_button = gr.Button("Clear Image") # Optional clear button

    with gr.Row():
        clear_chat_button = gr.Button("Clear Chat & Image")
        # submit_button = gr.Button("Send Message") # If we want an explicit send button

    # --- UI Update Logic ---
    def update_model_dropdown(provider_name):
        logging.info(f"Provider selected: {provider_name}")
        models = get_models_for_provider(provider_name)
        first_model = models[0] if models else None
        # Update visibility of refresh button
        refresh_btn_visibility = (provider_name == "ollama")
        # Update the selected_model_for_chat state when model dropdown changes
        # This will be set by the model_dropdown.change event too
        return gr.Dropdown(choices=models, value=first_model), gr.Button(visible=refresh_btn_visibility), first_model

    provider_dropdown.change(
        fn=update_model_dropdown, 
        inputs=[provider_dropdown],
        outputs=[model_dropdown, refresh_ollama_button, selected_model_for_chat] 
    )

    # When model_dropdown changes, update the selected_model_for_chat state
    def update_selected_model_state(model_name_from_dropdown):
        return model_name_from_dropdown

    model_dropdown.change(
        fn=update_selected_model_state,
        inputs=[model_dropdown],
        outputs=[selected_model_for_chat]
    )

    def refresh_ollama_models_and_update_ui(current_provider):
        logging.info(f"UI Refresh called for provider: {current_provider}")
        models = get_models_for_provider(current_provider) # This will use fallback if dynamic fails for ollama
        first_model = models[0] if models else None
        
        # If the current provider is ollama, this button is specifically for it.
        # If other providers were to have refresh, this logic might need adjustment.
        # For now, it correctly re-fetches (or gets fallback) for the current provider shown.
        if current_provider == "ollama":
            logging.info(f"Refreshing Ollama models list specifically.")
        return gr.Dropdown(choices=models, value=first_model), first_model

    refresh_ollama_button.click(
        fn=refresh_ollama_models_and_update_ui,
        inputs=[provider_dropdown],
        outputs=[model_dropdown, selected_model_for_chat]
    )

    # --- Submission Logic ---
    def wrapped_handle_submit(manual_model_str, text_msg, image_pil, chat_hist_state, current_selected_model_state):
        if not text_msg.strip() and image_pil is None:
            # Yield current state to avoid clearing chat for empty submit
            yield chat_hist_state, chat_hist_state, text_msg, image_pil # No change to textbox/image
            return
        
        final_model_to_use = manual_model_str.strip() if manual_model_str and manual_model_str.strip() else current_selected_model_state
        
        initial_state = list(chat_hist_state)
        final_chatbot_val, final_state_val = None, None
        
        # The generator yields (chatbot_display_history, new_state_history)
        for chatbot_val, state_val in chat_with_model(final_model_to_use, text_msg, image_pil, initial_state):
            final_chatbot_val = chatbot_val
            final_state_val = state_val
            # Keep text and image in place during streaming
            yield chatbot_val, state_val, text_msg, image_pil 
        
        # After streaming is done, clear the text and image inputs
        yield final_chatbot_val, final_state_val, "", None

    # Inputs for submit: manual model, text, image, history state, selected model from dropdowns
    submit_inputs = [
        manual_model_textbox, msg_textbox, image_upload, 
        state_history, selected_model_for_chat
    ]
    # Outputs for submit: chatbot, history state, text input, image input
    submit_outputs = [chatbot, state_history, msg_textbox, image_upload]

    msg_textbox.submit(fn=wrapped_handle_submit, inputs=submit_inputs, outputs=submit_outputs)
    # If using a submit_button:
    # submit_button.click(fn=wrapped_handle_submit, inputs=submit_inputs, outputs=submit_outputs)

    def clear_chat_and_image_func():
        logging.info("Chat and image cleared by user.")
        return [], [], "", None, None # Chatbot, state, msg_textbox, image_upload, selected_model_for_chat

    clear_chat_button.click(clear_chat_and_image_func, None, 
                            [chatbot, state_history, msg_textbox, image_upload, selected_model_for_chat], 
                            queue=False)
    
    # Initialize model dropdown for the default provider
    # This needs to happen after the UI components are defined
    def initial_load_models():
        default_provider = PROVIDER_NAMES[0] if PROVIDER_NAMES else None
        models, first_model, refresh_vis = [], None, False
        if default_provider:
            models = get_models_for_provider(default_provider)
            first_model = models[0] if models else None
            # Update refresh button visibility based on initial provider
            refresh_vis = (default_provider == "ollama")
        return gr.Dropdown(choices=models, value=first_model), first_model, gr.Button(visible=refresh_vis)

    demo.load(initial_load_models, outputs=[model_dropdown, selected_model_for_chat, refresh_ollama_button])

    gr.Markdown(
        "**Note:** For vision models, upload an image. For API-based providers (OpenAI, OpenRouter, etc.), "
        "ensure API keys (`OPENAI_API_KEY`, `OPENROUTER_API_KEY`, etc.) are set as environment variables. "
        "Refer to [LiteLLM Quick Start](https://docs.litellm.ai/docs/quick_start) for details."
    )

if __name__ == "__main__":
    logging.info("Starting Gradio application...")
    demo.launch() 