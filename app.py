import gradio as gr
from litellm import completion
import ollama
import logging
import base64
import io
from PIL import Image
import json
import re  # Add this at the top if not already present

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DEFAULT_ALT_TEXT_FOR_IMAGE_ONLY = "(Uploaded Image)" # Constant for default alt text

# --- Provider and Model Definitions ---
def get_ollama_models():
    try:
        models_data = ollama.list().get("models", [])
        model_names = []
        
        for m in models_data:
            # Handle different Ollama API response structures
            # Some versions use 'name', others might use 'model' or store it directly
            if isinstance(m, dict):
                if "name" in m:
                    model_names.append(m["name"])
                elif "model" in m:
                    model_names.append(m["model"])
                elif hasattr(m, "name") and m.name:
                    model_names.append(m.name)
                elif hasattr(m, "model") and m.model:
                    model_names.append(m.model)
        
        if not model_names:
            # If we still couldn't find models, try alternative API formats
            try:
                # Try direct model list (might be used in newer Ollama versions)
                all_models = ollama.list()
                if isinstance(all_models, list):
                    for m in all_models:
                        if isinstance(m, str):
                            model_names.append(m)
                        elif isinstance(m, dict) and "name" in m:
                            model_names.append(m["name"])
                        elif isinstance(m, dict) and "model" in m:
                            model_names.append(m["model"])
            except Exception as alt_e:
                logging.warning(f"Alternative Ollama model fetch failed: {alt_e}")
                
        logging.info(f"Successfully fetched Ollama models: {model_names}")
        return model_names
    except Exception as e:
        logging.error(f"Error fetching Ollama models: {e}", exc_info=True)
        
        # Try direct model fetching as fallback
        try:
            logging.info("Attempting direct Ollama model fetch")
            import requests
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                data = response.json()
                if "models" in data and isinstance(data["models"], list):
                    model_names = [m.get("name", m.get("model", "unknown")) for m in data["models"] if isinstance(m, dict)]
                    logging.info(f"Direct API fetch successful: {model_names}")
                    return model_names
        except Exception as fallback_e:
            logging.error(f"Direct API fallback also failed: {fallback_e}")
            
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

# Models that should display their thinking process
THINKING_MODELS = [
    "deepseek-r1:14b", "deepseek-r1:8b",  # Deepseek models often have thinking tags
    "think",  # Any model with "think" in the name
    "reasoning",  # Any model with "reasoning" in the name
    "cot"  # Chain of thought models
]

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

# Add this helper function
def clean_model_response(text, model_name=None):
    """
    Remove or preserve thinking tags and other content from model responses
    depending on the model type.
    
    Args:
        text: The text to clean
        model_name: Name of the model, to check if it's a thinking model
    
    Returns:
        Cleaned text with thinking tags preserved or removed as appropriate
    """
    if not text:
        return ""
    
    # Check if this is a "thinking model" that should display its reasoning
    is_thinking_model = False
    if model_name:
        # Extract base model name from fully qualified name (e.g., "ollama/deepseek-r1:14b" -> "deepseek-r1:14b")
        base_model = model_name.split('/')[-1] if '/' in model_name else model_name
        
        # Check if this model is in our thinking models list or contains keywords
        is_thinking_model = any(thinking_model in base_model.lower() 
                               for thinking_model in THINKING_MODELS) or \
                           any(keyword in base_model.lower() 
                               for keyword in ["think", "reasoning", "cot"])
    
    if is_thinking_model:
        # For thinking models, preserve the tags but format them nicely
        cleaned = text
        
        # Optional: Format <think> sections to look better
        cleaned = re.sub(r'<think>', '\n\n💭 *Thinking:*\n', cleaned)
        cleaned = re.sub(r'</think>', '\n\n', cleaned)
        
        # Format other tags but preserve content
        cleaned = re.sub(r'<reasoning>', '\n\n💡 *Reasoning:*\n', cleaned)
        cleaned = re.sub(r'</reasoning>', '\n\n', cleaned)
        cleaned = re.sub(r'<answer>', '\n\n✅ *Answer:*\n', cleaned)
        cleaned = re.sub(r'</answer>', '', cleaned)
    else:
        # For regular models, remove thinking sections entirely
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        cleaned = re.sub(r'<answer>|</answer>', '', cleaned)
        cleaned = re.sub(r'<reasoning>.*?</reasoning>', '', cleaned, flags=re.DOTALL)
    
    # Common cleaning for all models
    # Replace multiple newlines with just two
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    
    # Trim leading/trailing whitespace
    cleaned = cleaned.strip()
    
    return cleaned

# --- Chat Logic ---
def chat_with_model(selected_model_name, user_text, user_image_pil, history_state, system_prompt=None, temperature=0.7):
    history_for_chatbot_display = list(history_state)
    messages_for_llm = []

    # Add system prompt if provided
    if system_prompt and system_prompt.strip():
        messages_for_llm.append({"role": "system", "content": system_prompt.strip()})

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

    logging.info(f"Attempting to stream chat with model: {selected_model_name}, temperature: {temperature}")
    
    # Print the first few messages for debugging (excluding large base64 images)
    debug_messages = []
    for msg in messages_for_llm:
        if msg["role"] == "user" and isinstance(msg["content"], list):
            # For multipart messages, show only text parts or indicate image presence
            content_debug = []
            for part in msg["content"]:
                if part.get("type") == "text":
                    content_debug.append({"type": "text", "text": part["text"]})
                elif part.get("type") == "image_url":
                    content_debug.append({"type": "image_url", "note": "[IMAGE DATA]"})
            debug_msg = {"role": msg["role"], "content": content_debug}
        else:
            # For simple text messages, include directly
            debug_msg = msg
        debug_messages.append(debug_msg)
    
    logging.info(f"Messages for LiteLLM (debug view): {json.dumps(debug_messages, indent=2)}")

    if user_display_turn_item is not None:
        history_for_chatbot_display.append((user_display_turn_item, "")) # Placeholder for bot response
    else: # Should not happen if submit handler catches empty inputs
        yield history_for_chatbot_display, history_for_chatbot_display
        return
        
    try:
        # Attempt with streaming first
        streaming_enabled = True
        try:
            logging.info(f"Starting streaming request to {selected_model_name}")
            response_stream = completion(
                model=selected_model_name, 
                messages=messages_for_llm, 
                stream=True,
                temperature=float(temperature)
            )
            
            current_bot_response_text = ""
            chunks_received = 0
            empty_chunks = 0
            
            for chunk in response_stream:
                chunks_received += 1
                delta = None
                
                # Handle different response formats
                if hasattr(chunk, 'choices') and chunk.choices:
                    if hasattr(chunk.choices[0], 'delta'):
                        if hasattr(chunk.choices[0].delta, 'content'):
                            # Standard OpenAI-like format
                            delta = chunk.choices[0].delta.content
                        elif hasattr(chunk.choices[0].delta, 'text'):
                            # Alternative format seen in some models
                            delta = chunk.choices[0].delta.text
                    elif hasattr(chunk.choices[0], 'text'):
                        # Direct text format
                        delta = chunk.choices[0].text
                    elif hasattr(chunk.choices[0], 'message') and hasattr(chunk.choices[0].message, 'content'):
                        # Complete message format
                        delta = chunk.choices[0].message.content
                
                if delta:
                    logging.debug(f"Received chunk {chunks_received}: '{delta}'")
                    current_bot_response_text += delta
                    
                    # Clean the current accumulated text before displaying, passing model name
                    cleaned_text = clean_model_response(current_bot_response_text, selected_model_name)
                    
                    # Only update the UI if there's clean content to show
                    if cleaned_text:
                        history_for_chatbot_display[-1] = (user_display_turn_item, cleaned_text)
                        yield history_for_chatbot_display, history_for_chatbot_display
                else:
                    empty_chunks += 1
                    logging.debug(f"Received empty chunk {chunks_received}")
            
            # Clean the final response with model name
            final_cleaned_text = clean_model_response(current_bot_response_text, selected_model_name)
            
            # If we got chunks but text is still empty, use non-streaming as fallback
            if chunks_received > 0 and not final_cleaned_text.strip():
                logging.warning(f"Received {chunks_received} chunks but no clean text content from {selected_model_name}")
                logging.warning(f"Raw text before cleaning: '{current_bot_response_text[:100]}...'")
                logging.warning(f"Empty chunks: {empty_chunks}, Non-empty chunks: {chunks_received - empty_chunks}")
                logging.warning("Will try non-streaming mode as fallback")
                streaming_enabled = False
            elif chunks_received == 0:
                # No chunks received despite no errors - fall back to non-streaming
                logging.warning(f"No chunks received from {selected_model_name} in streaming mode. Falling back to non-streaming.")
                streaming_enabled = False
            else:
                logging.info(f"Successfully streamed response from {selected_model_name} ({chunks_received} chunks, {empty_chunks} empty)")
                
                # Ensure the final cleaned response is in the UI
                if final_cleaned_text.strip():
                    logging.info(f"Final cleaned response text: '{final_cleaned_text[:100]}...'")
                    history_for_chatbot_display[-1] = (user_display_turn_item, final_cleaned_text)
                    yield history_for_chatbot_display, history_for_chatbot_display
                    # If we successfully got a response, return
                    return
                else:
                    # If we got through streaming but cleaned text is empty, try non-streaming
                    logging.warning("Cleaned streamed response is empty, trying non-streaming mode")
                    streaming_enabled = False
                 
        except Exception as stream_err:
            logging.warning(f"Error in streaming mode with {selected_model_name}: {stream_err}. Falling back to non-streaming.")
            streaming_enabled = False
        
        # If streaming failed or produced no chunks, try non-streaming mode
        if not streaming_enabled:
            logging.info(f"Attempting non-streaming completion with {selected_model_name}")
            response = completion(
                model=selected_model_name, 
                messages=messages_for_llm, 
                stream=False,
                temperature=float(temperature)
            )
            
            logging.info(f"Non-streaming response type: {type(response)}")
            if hasattr(response, '__dict__'):
                logging.info(f"Response attributes: {dir(response)}")
            
            response_text = None
            
            # Try different response formats
            if response:
                if hasattr(response, 'choices') and response.choices:
                    if hasattr(response.choices[0], 'message'):
                        response_text = response.choices[0].message.content
                    elif hasattr(response.choices[0], 'text'):
                        response_text = response.choices[0].text
                elif hasattr(response, 'content'):
                    response_text = response.content
                elif hasattr(response, 'text'):
                    response_text = response.text
            
            if response_text:
                # Clean the response text with model name
                cleaned_response = clean_model_response(response_text, selected_model_name)
                logging.info(f"Successfully received non-streaming response: '{cleaned_response[:100]}...'")
                
                if cleaned_response.strip():
                    history_for_chatbot_display[-1] = (user_display_turn_item, cleaned_response)
                    yield history_for_chatbot_display, history_for_chatbot_display
                    return
                else:
                    logging.error(f"Non-streaming response produced empty text after cleaning. Raw: '{response_text[:100]}...'")
                    raise Exception("Model returned content that was filtered out entirely")
            else:
                logging.error(f"No text content found in non-streaming response")
                raise Exception("Model returned no valid response content in non-streaming mode")
            
    except Exception as e:
        logging.error(f"Error during model interaction with {selected_model_name}: {e}", exc_info=True)
        error_detail = f"Error interacting with model {selected_model_name}: {str(e)}\n\nCommon issues:\n"
        error_detail += "- Ensure the model name is correct (e.g., 'ollama/mistral', 'openrouter/openai/gpt-4o').\n"
        error_detail += "- For Ollama models, make sure you've downloaded the model with 'ollama pull MODEL_NAME'.\n"
        error_detail += "- For API-based models, ensure API keys (e.g., OPENAI_API_KEY, OPENROUTER_API_KEY) are set as environment variables.\n"
        error_detail += "- Check if the model server (e.g., Ollama for local models) is running and accessible.\n"
        error_detail += "- The model might be overloaded or unavailable, or not support the input type (e.g. vision for text-only model)."
        history_for_chatbot_display[-1] = (user_display_turn_item, error_detail)
        yield history_for_chatbot_display, history_for_chatbot_display

# --- JavaScript for Features ---
ui_enhancement_js = """
function setupChatEnhancements() {
    console.log("Setting up chat enhancements...");
    
    // Function to auto-scroll chat to bottom
    function autoScrollChat() {
        const chatContainer = document.querySelector('.chatbot');
        if (chatContainer) {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
    }

    // Function to add copy buttons to messages
    function addCopyButtons() {
        document.querySelectorAll('.chatbot .message').forEach(msg => {
            // Check if this message already has a copy button
            if (!msg.querySelector('.copy-button')) {
                // Create and append copy button
                const btn = document.createElement('button');
                btn.className = 'copy-button';
                btn.innerHTML = 'Copy';
                btn.onclick = function(e) {
                    e.stopPropagation();
                    const text = msg.querySelector('.message-text') ? 
                                 msg.querySelector('.message-text').innerText : 
                                 msg.innerText;
                    navigator.clipboard.writeText(text).then(() => {
                        btn.innerHTML = 'Copied!';
                        setTimeout(() => { btn.innerHTML = 'Copy'; }, 2000);
                    });
                };
                msg.style.position = 'relative';
                msg.appendChild(btn);
            }
        });
    }

    // Function to add timestamps to messages
    function addTimestamps() {
        document.querySelectorAll('.chatbot .message').forEach(msg => {
            // Check if this message already has a timestamp
            if (!msg.querySelector('.message-timestamp')) {
                // Create and append timestamp
                const timestamp = document.createElement('div');
                timestamp.className = 'message-timestamp';
                timestamp.innerText = new Date().toLocaleTimeString();
                msg.appendChild(timestamp);
            }
        });
    }
    
    // Apply enhancements immediately
    autoScrollChat();
    addCopyButtons();
    addTimestamps();
    
    // Set up mutation observer to watch for changes in chat
    const observer = new MutationObserver(() => {
        console.log("Chat content changed!");
        autoScrollChat();
        addCopyButtons();
        addTimestamps();
    });
    
    // Start observing
    const chatContainer = document.querySelector('.chatbot');
    if (chatContainer) {
        observer.observe(chatContainer, { childList: true, subtree: true });
        console.log("Observer attached to chat container");
    } else {
        console.log("Chat container not found!");
    }
    
    // Auto-scroll on window resize (helps with mobile)
    window.addEventListener('resize', autoScrollChat);
    
    return true;
}

// Add styles directly to the document
function addCustomStyles() {
    const styleElement = document.createElement('style');
    styleElement.textContent = `
        /* Timestamp styling */
        .message-timestamp {
            font-size: 0.7em;
            opacity: 0.7;
            margin-top: 2px;
        }
        
        /* Typing indicator */
        .typing-indicator {
            display: inline-block;
            padding: 6px 12px;
            background-color: var(--color-accent-soft, #f0f0f0);
            border-radius: 12px;
            margin: 5px 0;
        }
        .typing-indicator span {
            display: inline-block;
            width: 8px;
            height: 8px;
            background-color: rgba(128, 128, 128, 0.7);
            border-radius: 50%;
            margin-right: 3px;
            animation: typingBounce 1.2s infinite ease-in-out;
        }
        .typing-indicator span:nth-child(2) {
            animation-delay: 0.2s;
        }
        .typing-indicator span:nth-child(3) {
            animation-delay: 0.4s;
            margin-right: 0;
        }
        @keyframes typingBounce {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-4px); }
        }
        
        /* Copy button for messages */
        .copy-button {
            opacity: 0;
            transition: opacity 0.3s;
            position: absolute;
            top: 5px;
            right: 5px;
            padding: 4px 8px;
            background: var(--background-fill-secondary, #f7f7f7);
            border: 1px solid var(--border-color-primary, #ddd);
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
        }
        .chatbot .message:hover .copy-button {
            opacity: 1;
        }
    `;
    document.head.appendChild(styleElement);
    return true;
}

// Initialize on page load and after Gradio updates
function initializeEnhancements() {
    // First add the styles
    addCustomStyles();
    
    // Check if Gradio has fully loaded
    if (document.readyState === 'complete') {
        setupChatEnhancements();
    } else {
        // Try immediately anyway, then after a delay
        setTimeout(setupChatEnhancements, 100);
        setTimeout(setupChatEnhancements, 500);
        setTimeout(setupChatEnhancements, 1000);
        setTimeout(setupChatEnhancements, 2000);
    }
    
    // Also run when Gradio updates its DOM
    const gradioContainer = document.querySelector('#component-0');
    if (gradioContainer) {
        const gradioObserver = new MutationObserver(() => {
            setTimeout(setupChatEnhancements, 100);
        });
        gradioObserver.observe(gradioContainer, { childList: true, subtree: true });
    }
    
    return true;
}

// Start initializing
initializeEnhancements();
"""

# CSS for basic UI elements - still needed
custom_css = """
.dark-mode {
    --background-fill-primary: #1f1f1f !important;
    --background-fill-secondary: #2b2b2b !important;
    --text-color: #ffffff !important;
    --color-accent-soft: #3a3a3a !important;
    --border-color-primary: #444444 !important;
}

.light-mode {
    --background-fill-primary: #ffffff !important;
    --background-fill-secondary: #f7f7f7 !important;
    --text-color: #000000 !important;
    --color-accent-soft: #f0f0f0 !important;
    --border-color-primary: #ddd !important;
}

.chat-interface {
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.model-selection {
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    padding-bottom: 10px !important;
}

.input-row {
    margin-top: 10px !important;
}

.chat-header {
    margin-bottom: 10px !important;
    font-weight: 600 !important;
}

.system-prompt {
    border-radius: 8px;
    padding: 10px;
    margin-bottom: 10px;
}

.temperature-slider .gr-slider {
    margin-top: 5px !important;
}

.model-info-row {
    font-size: 0.85em;
    margin-top: 8px;
}

.app-title {
    font-weight: 600 !important;
    margin-bottom: 20px !important;
}
"""

# --- Gradio UI ---    
with gr.Blocks(theme=gr.themes.Soft(), title="LiteLLM Chat UI", css=custom_css) as demo:
    dark_mode = gr.State(False)
    
    # Create a hidden component to store JavaScript updates
    js_updates = gr.HTML("", visible=False, elem_id="js-container")
    
    # Add a dedicated script component for UI enhancements
    ui_script = gr.HTML(f"<script>{ui_enhancement_js}</script>", visible=False)
    
    def toggle_dark_mode(current_mode):
        new_mode = not current_mode
        js_code = ""
        if new_mode:
            # Switching to dark mode
            js_code = """
                document.querySelector('body').classList.add('dark-mode');
                document.querySelector('body').classList.remove('light-mode');
            """
        else:
            # Switching to light mode
            js_code = """
                document.querySelector('body').classList.add('light-mode');
                document.querySelector('body').classList.remove('dark-mode');
            """
        return new_mode, f"""
            <script>
                {js_code}
                // Reapply enhancements after theme change
                setTimeout(setupChatEnhancements, 100);
            </script>
        """
    
    gr.Markdown("# 🤖 LiteLLM & Gradio Chat Interface", elem_classes=["app-title"])
    
    with gr.Row():
        theme_btn = gr.Button("🌓 Toggle Dark Mode", scale=1, variant="secondary")
    
    state_history = gr.State([])
    # Hidden state to store the fully qualified model name from dropdowns or manual input
    selected_model_for_chat = gr.State(None)
    # State to track if a response is being generated
    is_generating = gr.State(False)

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group(elem_classes=["model-selection"]):
                gr.Markdown("### Model Selection", elem_classes=["chat-header"])
                
                provider_dropdown = gr.Dropdown(
                    label="Provider", 
                    choices=PROVIDER_NAMES, 
                    value=PROVIDER_NAMES[0] if PROVIDER_NAMES else None,
                    scale=1
                )
                model_dropdown = gr.Dropdown(
                    label="Model", 
                    choices=[], # Will be populated based on provider
                    scale=1,
                    interactive=True
                )
                manual_model_textbox = gr.Textbox(
                    label="Manual Model String",
                    placeholder="e.g., openrouter/google/gemini-pro",
                    scale=1
                )
                refresh_ollama_button = gr.Button("🔄 Refresh Ollama Models", scale=1, variant="secondary", visible=(PROVIDER_NAMES[0] == "ollama" if PROVIDER_NAMES else False))
                
                with gr.Accordion("Generation Settings", open=True):
                    system_prompt = gr.Textbox(
                        label="System Prompt",
                        placeholder="Instructions for the AI...",
                        lines=2,
                        elem_classes=["system-prompt"]
                    )
                    temperature = gr.Slider(
                        label="Temperature",
                        minimum=0.0,
                        maximum=2.0,
                        value=0.7,
                        step=0.1,
                        elem_classes=["temperature-slider"]
                    )

        with gr.Column(scale=2):
            with gr.Group(elem_classes=["chat-interface"]):
                gr.Markdown("### Chat Window", elem_classes=["chat-header"])
                
                # Chat display
                chatbot = gr.Chatbot(
                    label="Chat Conversation", 
                    height=550, 
                    bubble_full_width=False, 
                    show_label=False,
                    avatar_images=("👤", "🤖"),
                    elem_id="chat-window"
                )
                
                # Typing indicator element
                typing_indicator = gr.HTML(
                    """
                    <div class="typing-indicator" style="display: none;">
                        <span></span><span></span><span></span>
                    </div>
                    """, 
                    visible=True,
                    elem_id="typing-indicator"
                )
                
                # Input area
                with gr.Group():
                    with gr.Row(elem_classes=["input-row"]):
                        msg_textbox = gr.Textbox(
                            label="Your Message", 
                            placeholder="Type your message here...", 
                            show_label=False, 
                            scale=3,
                            container=False
                        )
                        image_upload = gr.Image(
                            type="pil", 
                            label="Image", 
                            height=80, 
                            show_label=False, 
                            scale=1,
                            sources=["upload", "clipboard"]
                        )
                    
                    with gr.Row():
                        clear_chat_button = gr.Button("🗑️ Clear Chat", scale=1, variant="secondary")
                        send_button = gr.Button("📤 Send", scale=1, variant="primary")

                # Model info footer
                with gr.Row(elem_classes=["model-info-row"]):
                    gr.Markdown(
                        "📊 **Current Model:** <span id='current-model'>None selected</span> | "
                        "🔥 **Temperature:** <span id='current-temp'>0.7</span>",
                        elem_id="model-info-display"
                    )

    with gr.Accordion("API Key & Usage Notes", open=False):
        gr.Markdown(
            "**API Keys:** For providers like OpenAI, OpenRouter, Groq, Anthropic, etc., ensure you have the respective API keys "
            "set as environment variables (e.g., `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `GROQ_API_KEY`).\n\n"
            "**Vision Models:** To use vision capabilities, upload an image along with your text. Ensure the selected model supports vision input.\n\n"
            "**Manual Model String:** Use the manual input field for models not listed in the dropdowns. Prefix appropriately (e.g., `ollama/llava`, `openrouter/anthropic/claude-3-haiku`).\n\n"
            "Refer to [LiteLLM Documentation](https://docs.litellm.ai/docs/providers) for more details on providers and model names."
        )

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
    def wrapped_handle_submit(manual_model_str, text_msg, image_pil, chat_hist_state, current_selected_model_state, sys_prompt, temp, is_generating_state):
        if is_generating_state:
            # If already generating, don't allow a new submission
            yield chat_hist_state, chat_hist_state, text_msg, image_pil, "", True
            return
            
        if not text_msg.strip() and image_pil is None:
            # Yield current state to avoid clearing chat for empty submit
            yield chat_hist_state, chat_hist_state, text_msg, image_pil, "", False
            return
        
        # Set generating state to true
        is_generating_state = True
        
        final_model_to_use = manual_model_str.strip() if manual_model_str and manual_model_str.strip() else current_selected_model_state
        
        initial_state = list(chat_hist_state)
        final_chatbot_val, final_state_val = None, None
        
        # Create JavaScript to update UI display of current model and temperature
        # and show typing indicator
        js_code = f"""
        <script>
            (function() {{
                if (document.getElementById('current-model')) {{
                    document.getElementById('current-model').textContent = "{final_model_to_use if final_model_to_use else 'None'}";
                }}
                if (document.getElementById('current-temp')) {{
                    document.getElementById('current-temp').textContent = "{temp}";
                }}
                
                // Show typing indicator
                const indicator = document.querySelector('.typing-indicator');
                if (indicator) {{
                    indicator.style.display = 'inline-block';
                }}
                
                // Force a scroll to bottom
                setTimeout(() => {{
                    const chatContainer = document.querySelector('.chatbot');
                    if (chatContainer) {{
                        chatContainer.scrollTop = chatContainer.scrollHeight;
                    }}
                }}, 100);
                
                // Call our enhancement functions
                if (typeof setupChatEnhancements === 'function') {{
                    setupChatEnhancements();
                }}
            }})();
        </script>
        """
        
        # The generator yields (chatbot_display_history, new_state_history)
        for chatbot_val, state_val in chat_with_model(final_model_to_use, text_msg, image_pil, initial_state, sys_prompt, temp):
            final_chatbot_val = chatbot_val
            final_state_val = state_val
            # Keep text and image in place during streaming
            yield chatbot_val, state_val, text_msg, image_pil, js_code, True
        
        # Hide typing indicator when done
        js_code_final = f"""
        <script>
            (function() {{
                if (document.getElementById('current-model')) {{
                    document.getElementById('current-model').textContent = "{final_model_to_use if final_model_to_use else 'None'}";
                }}
                if (document.getElementById('current-temp')) {{
                    document.getElementById('current-temp').textContent = "{temp}";
                }}
                
                // Hide typing indicator
                const indicator = document.querySelector('.typing-indicator');
                if (indicator) {{
                    indicator.style.display = 'none';
                }}
                
                // Force a scroll to bottom and refresh enhancements
                setTimeout(() => {{
                    if (typeof setupChatEnhancements === 'function') {{
                        setupChatEnhancements();
                    }}
                }}, 100);
            }})();
        </script>
        """
        
        # After streaming is done, clear the text and image inputs
        yield final_chatbot_val, final_state_val, "", None, js_code_final, False

    # Inputs for submit: manual model, text, image, history state, selected model from dropdowns, system prompt, temperature, is_generating
    submit_inputs = [
        manual_model_textbox, msg_textbox, image_upload, 
        state_history, selected_model_for_chat, system_prompt, temperature, is_generating
    ]
    # Outputs for submit: chatbot, history state, text input, image input, js_updates, is_generating
    submit_outputs = [chatbot, state_history, msg_textbox, image_upload, js_updates, is_generating]

    msg_textbox.submit(fn=wrapped_handle_submit, inputs=submit_inputs, outputs=submit_outputs)
    send_button.click(fn=wrapped_handle_submit, inputs=submit_inputs, outputs=submit_outputs)

    def clear_chat_and_image_func():
        logging.info("Chat and image cleared by user.")
        js_code = """
        <script>
            (function() {
                if (document.getElementById('current-model')) {
                    document.getElementById('current-model').textContent = "None selected";
                }
                
                // Hide typing indicator
                const indicator = document.querySelector('.typing-indicator');
                if (indicator) {
                    indicator.style.display = 'none';
                }
                
                // Clear chat UI
                setTimeout(() => {
                    if (typeof setupChatEnhancements === 'function') {
                        setupChatEnhancements();
                    }
                }, 100);
            })();
        </script>
        """
        return [], [], "", None, None, False, js_code # Chatbot, state, msg_textbox, image_upload, selected_model_for_chat, is_generating, js_updates

    clear_chat_button.click(clear_chat_and_image_func, None, 
                            [chatbot, state_history, msg_textbox, image_upload, selected_model_for_chat, is_generating, js_updates], 
                            queue=False)
    
    # Handle dark mode toggle
    theme_btn.click(
        fn=toggle_dark_mode,
        inputs=[dark_mode],
        outputs=[dark_mode, js_updates]
    )
    
    # Initialize theme to match initial state
    demo.load(
        fn=lambda: f"""
        <script>
            document.querySelector('body').classList.add('light-mode');
            // Setup enhancements will be handled by the dedicated script component
        </script>
        """,
        inputs=None,
        outputs=js_updates,
    )
    
    # Initialize model dropdown for the default provider
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

if __name__ == "__main__":
    logging.info("Starting Gradio application...")
    demo.launch() 