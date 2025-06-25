import multiprocessing
import ctypes

from huggingface_hub import hf_hub_download

import pytest

import llama_cpp

@pytest.fixture
def mmproj_model_path():
    repo_id = "second-state/Llava-v1.5-7B-GGUF"
    filename = "llava-v1.5-7b-mmproj-model-f16.gguf"
    model_path = hf_hub_download(repo_id, filename)
    return model_path

@pytest.fixture
def llava_cpp_model_path():
    repo_id = "second-state/Llava-v1.5-7B-GGUF"
    filename = "llava-v1.5-7b-Q8_0.gguf"
    model_path = hf_hub_download(repo_id, filename)
    return model_path

def test_real_llava(llava_cpp_model_path, mmproj_model_path):
    print("initializing model")
    model = llama_cpp.Llama(
        llava_cpp_model_path,
        n_ctx=2048,
        n_batch=512,
        n_threads=multiprocessing.cpu_count(),
        n_threads_batch=multiprocessing.cpu_count(),
        logits_all=False,
        verbose=False,
    )

    # Initialize the LLaVA chat handler
    from llama_cpp.llama_chat_format import Llava15ChatHandler
    print("initializing chat handler")
    chat_handler = Llava15ChatHandler(clip_model_path=mmproj_model_path, llama_model=model)

    # Create a chat message with the image
    print("creating chat message")
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": "./tests/monalisa.jpg"
                },
                {
                    "type": "text",
                    "text": "Do you know who drew this painting?"
                }
            ]
        }
    ]

    # Generate response
    print("generating response")
    response = chat_handler(
        llama=model,
        messages=messages,
        max_tokens=200,
        temperature=0.2,
        top_p=0.95,
        stream=False
    )

    print("response", response)
    # Check that we got a response
    assert response is not None
    assert "choices" in response
    assert len(response["choices"]) > 0
    assert "message" in response["choices"][0]
    assert "content" in response["choices"][0]["message"]
    
    # The response should mention Leonardo da Vinci
    content = response["choices"][0]["message"]["content"].lower()
    assert "leonardo" in content and "vinci" in content  # Artist name should be in response
