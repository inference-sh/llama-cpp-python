from __future__ import annotations

import os
import sys
import json
import ctypes
import dataclasses
import random
import string
import warnings

from datetime import datetime
from contextlib import ExitStack
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    Protocol,
    cast,
)

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment

import numpy as np
import numpy.typing as npt

import llama_cpp.llama_cpp as llama_cpp
import llama_cpp.llama as llama
import llama_cpp.llama_types as llama_types
import llama_cpp.llama_grammar as llama_grammar

from ._logger import logger
from ._utils import suppress_stdout_stderr, Singleton

### Chat Completion Handler ###


class LlamaChatCompletionHandler(Protocol):
    """Base Protocol for a llama chat completion handler.

    Very generic protocol that can be used to implement any chat format.
    The only hard requirement is that it must return a ChatCompletion when
    stream=False and an iterator of ChatCompletionChunks when stream=True."""

    def __call__(
        self,
        *,
        # llama.cpp instance
        llama: llama.Llama,
        # openai api parameters
        messages: List[llama_types.ChatCompletionRequestMessage],
        functions: Optional[List[llama_types.ChatCompletionFunction]] = None,
        function_call: Optional[llama_types.ChatCompletionRequestFunctionCall] = None,
        tools: Optional[List[llama_types.ChatCompletionTool]] = None,
        tool_choice: Optional[llama_types.ChatCompletionToolChoiceOption] = None,
        temperature: float = 0.2,
        top_p: float = 0.95,
        top_k: int = 40,
        stream: bool = False,
        stop: Optional[Union[str, List[str]]] = [],
        seed: Optional[int] = None,
        response_format: Optional[
            llama_types.ChatCompletionRequestResponseFormat
        ] = None,
        max_tokens: Optional[int] = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        repeat_penalty: float = 1.1,
        model: Optional[str] = None,
        logit_bias: Optional[Dict[str, float]] = None,
        # llama.cpp parameters
        min_p: float = 0.05,
        typical_p: float = 1.0,
        tfs_z: float = 1.0,
        mirostat_mode: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
        logits_processor: Optional[llama.LogitsProcessorList] = None,
        grammar: Optional[llama.LlamaGrammar] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        **kwargs,  # type: ignore
    ) -> Union[
        llama_types.CreateChatCompletionResponse,
        Iterator[llama_types.CreateChatCompletionStreamResponse],
    ]: ...


class LlamaChatCompletionHandlerNotFoundException(Exception):
    pass


class LlamaChatCompletionHandlerRegistry(Singleton):
    _chat_handlers: Dict[str, LlamaChatCompletionHandler] = {}

    def register_chat_completion_handler(
        self,
        name: str,
        chat_handler: LlamaChatCompletionHandler,
        overwrite: bool = False,
    ):
        if not overwrite and name in self._chat_handlers:
            raise ValueError(
                f"Formatter with name '{name}' is already registered. Use `overwrite=True` to overwrite it."
            )
        self._chat_handlers[name] = chat_handler

    def unregister_chat_handler(self, name: str):
        if name in self._chat_handlers:
            del self._chat_handlers[name]
        else:
            raise ValueError(f"No formatter registered under the name '{name}'.")

    def get_chat_completion_handler_by_name(
        self, name: str
    ) -> LlamaChatCompletionHandler:
        try:
            chat_handler = self._chat_handlers[name]
            return chat_handler
        except KeyError:
            raise LlamaChatCompletionHandlerNotFoundException(
                f"Invalid chat handler: {name} (valid formats: {list(self._chat_handlers.keys())})"
            )


def get_chat_completion_handler(name: str) -> LlamaChatCompletionHandler:
    return LlamaChatCompletionHandlerRegistry().get_chat_completion_handler_by_name(
        name
    )


def register_chat_completion_handler(name: str):
    def decorator(f: LlamaChatCompletionHandler):
        LlamaChatCompletionHandlerRegistry().register_chat_completion_handler(name, f)
        return f

    return decorator


### Chat Formatter ###


@dataclasses.dataclass
class ChatFormatterResponse:
    """Dataclass that stores completion parameters for a given chat format and
    create_chat_completion request.

    prompt contains the formatted prompt generated from the chat format and messages.
    stop contains the stop token or list of stop tokens to use for the chat format."""

    prompt: str
    stop: Optional[Union[str, List[str]]] = None
    stopping_criteria: Optional[llama.StoppingCriteriaList] = None
    added_special: bool = False


class ChatFormatter(Protocol):
    """Base Protocol for a chat formatter. A chat formatter is a function that
    takes a list of messages and returns a chat format response which can be used
    to generate a completion. The response can also include a stop token or list
    of stop tokens to use for the completion."""

    def __call__(
        self,
        *,
        messages: List[llama_types.ChatCompletionRequestMessage],
        **kwargs: Any,
    ) -> ChatFormatterResponse: ...


class Jinja2ChatFormatter(ChatFormatter):
    def __init__(
        self,
        template: str,
        eos_token: str,
        bos_token: str,
        add_generation_prompt: bool = True,
        stop_token_ids: Optional[List[int]] = None,
    ):
        """A chat formatter that uses jinja2 templates to format the prompt."""
        self.template = template
        self.eos_token = eos_token
        self.bos_token = bos_token
        self.add_generation_prompt = add_generation_prompt
        self.stop_token_ids = (
            set(stop_token_ids) if stop_token_ids is not None else None
        )

        self._environment = ImmutableSandboxedEnvironment(
            loader=jinja2.BaseLoader(),
            trim_blocks=True,
            lstrip_blocks=True,
        ).from_string(self.template)

    @staticmethod
    def strftime_now(f: str) -> str:
        return datetime.now().strftime(f)

    def __call__(
        self,
        *,
        messages: List[llama_types.ChatCompletionRequestMessage],
        functions: Optional[List[llama_types.ChatCompletionFunction]] = None,
        function_call: Optional[llama_types.ChatCompletionRequestFunctionCall] = None,
        tools: Optional[List[llama_types.ChatCompletionTool]] = None,
        tool_choice: Optional[llama_types.ChatCompletionToolChoiceOption] = None,
        **kwargs: Any,
    ) -> ChatFormatterResponse:
        def raise_exception(message: str):
            raise ValueError(message)

        prompt = self._environment.render(
            messages=messages,
            eos_token=self.eos_token,
            bos_token=self.bos_token,
            raise_exception=raise_exception,
            add_generation_prompt=self.add_generation_prompt,
            functions=functions,
            function_call=function_call,
            tools=tools,
            tool_choice=tool_choice,
            strftime_now=self.strftime_now,
        )

        stopping_criteria = None
        if self.stop_token_ids is not None:

            def stop_on_last_token(
                tokens: npt.NDArray[np.intc], logits: npt.NDArray[np.single]
            ) -> bool:
                return tokens[-1] in self.stop_token_ids

            stopping_criteria = llama.StoppingCriteriaList([stop_on_last_token])

        return ChatFormatterResponse(
            prompt=prompt,
            stop=[self.eos_token],
            stopping_criteria=stopping_criteria,
            added_special=True,
        )

    def to_chat_handler(self) -> LlamaChatCompletionHandler:
        return chat_formatter_to_chat_completion_handler(self)

def chat_formatter_to_chat_completion_handler(
    chat_formatter: ChatFormatter,
) -> LlamaChatCompletionHandler:
    def chat_completion_handler(
        *,
        llama: llama.Llama,
        messages: List[llama_types.ChatCompletionRequestMessage],
        functions: Optional[List[llama_types.ChatCompletionFunction]] = None,
        function_call: Optional[llama_types.ChatCompletionRequestFunctionCall] = None,
        tools: Optional[List[llama_types.ChatCompletionTool]] = None,
        tool_choice: Optional[llama_types.ChatCompletionToolChoiceOption] = None,
        temperature: float = 0.2,
        top_p: float = 0.95,
        top_k: int = 40,
        min_p: float = 0.05,
        typical_p: float = 1.0,
        stream: bool = False,
        stop: Optional[Union[str, List[str]]] = [],
        seed: Optional[int] = None,
        response_format: Optional[
            llama_types.ChatCompletionRequestResponseFormat
        ] = None,
        max_tokens: Optional[int] = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        repeat_penalty: float = 1.1,
        tfs_z: float = 1.0,
        mirostat_mode: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
        model: Optional[str] = None,
        logits_processor: Optional[llama.LogitsProcessorList] = None,
        grammar: Optional[llama.LlamaGrammar] = None,
        logit_bias: Optional[Dict[str, float]] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        **kwargs,  # type: ignore
    ) -> Union[
        llama_types.CreateChatCompletionResponse,
        Iterator[llama_types.CreateChatCompletionStreamResponse],
    ]:
        result = chat_formatter(
            messages=messages,
            functions=functions,
            function_call=function_call,
            tools=tools,
            tool_choice=tool_choice,
        )


        prompt = llama.tokenize(
            result.prompt.encode("utf-8"),
            add_bos=not result.added_special,
            special=True,
        )
        if result.stop is not None:
            stop = [] if stop is None else [stop] if isinstance(stop, str) else stop
            rstop = result.stop if isinstance(result.stop, list) else [result.stop]
            stop = stop + rstop

        stopping_criteria = None
        if result.stopping_criteria is not None:
            stopping_criteria = result.stopping_criteria

        if response_format is not None and response_format["type"] == "json_object":
            grammar = _grammar_for_response_format(
                response_format, verbose=llama.verbose
            )

        # Convert legacy functions to tools
        if functions is not None:
            tools = [
                {
                    "type": "function",
                    "function": function,
                }
                for function in functions
            ]

        # Convert legacy function_call to tool_choice
        if function_call is not None:
            if isinstance(function_call, str) and (
                function_call == "none" or function_call == "auto"
            ):
                tool_choice = function_call
            if isinstance(function_call, dict) and "name" in function_call:
                tool_choice = {
                    "type": "function",
                    "function": {
                        "name": function_call["name"],
                    },
                }

        tool = None
        if (
            tool_choice is not None
            and isinstance(tool_choice, dict)
            and tools is not None
        ):
            name = tool_choice["function"]["name"]
            tool = next((t for t in tools if t["function"]["name"] == name), None)
            if tool is None:
                raise ValueError(f"Tool choice '{name}' not found in tools.")
            schema = tool["function"]["parameters"]
            try:
                # create grammar from json schema
                grammar = llama_grammar.LlamaGrammar.from_json_schema(
                    json.dumps(schema), verbose=llama.verbose
                )
            except Exception as e:
                if llama.verbose:
                    print(str(e), file=sys.stderr)
                grammar = llama_grammar.LlamaGrammar.from_string(
                    llama_grammar.JSON_GBNF, verbose=llama.verbose
                )

        # Handle auto tool choice specially to detect function calls
        if tool_choice == "auto" and tools is not None and len(tools) > 0:
            # First generate normally until we hit a tool call tag
            completion_or_chunks = llama.create_completion(
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                typical_p=typical_p,
                logprobs=top_logprobs if logprobs else None,
                stream=stream,
                stop=["<tool_call>"],  # Stop at tool call start tag
                seed=seed,
                max_tokens=max_tokens,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                repeat_penalty=repeat_penalty,
                tfs_z=tfs_z,
                mirostat_mode=mirostat_mode,
                mirostat_tau=mirostat_tau,
                mirostat_eta=mirostat_eta,
                model=model,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                grammar=grammar,
                logit_bias=logit_bias,
            )
            
            # If we hit a tool call tag, create grammar and continue with tool call
            if not stream:
                completion = cast(llama_types.CreateCompletionResponse, completion_or_chunks)
                response_text = completion["choices"][0]["text"]
                
                if response_text.rstrip().endswith("<tool_call>"):
                    if llama.verbose:
                        print("[DEBUG] Found tool call tag, switching to tool grammar", file=sys.stderr)
                    
                    # Create tool call grammar based on available tools
                    function_names = " | ".join([f'''"functions.{t["function"]["name"]}:"''' for t in tools])
                    tool_call_gbnf = (
                        'root ::= "<tool_call>" "\\n" functions\n'
                        f"functions ::= {function_names}\n"
                    )
                    
                    # First get the tool name
                    try:
                        name_grammar = llama_grammar.LlamaGrammar.from_string(
                            tool_call_gbnf,
                            verbose=llama.verbose
                        )
                        
                        # Generate tool name
                        tool_name_completion = llama.create_completion(
                            prompt=prompt + response_text,
                            temperature=0,
                            stream=False,
                            stop=[":"],
                            max_tokens=None,
                            grammar=name_grammar,
                        )
                        
                        # Get the selected tool
                        tool_name = tool_name_completion["choices"][0]["text"].split("\n")[-1][len("functions."):-1]
                        tool = next((t for t in tools if t["function"]["name"] == tool_name), None)
                        
                        if tool:
                            # Create grammar for tool parameters
                            try:
                                tool_grammar = llama_grammar.LlamaGrammar.from_json_schema(
                                    json.dumps(tool["function"]["parameters"]),
                                    verbose=llama.verbose
                                )
                            except Exception as e:
                                if llama.verbose:
                                    print(f"[DEBUG] Failed to parse function parameters as JSON schema: {e}", file=sys.stderr)
                                tool_grammar = llama_grammar.LlamaGrammar.from_string(
                                    llama_grammar.JSON_GBNF,
                                    verbose=llama.verbose
                                )
                        else:
                            if llama.verbose:
                                print(f"[DEBUG] Tool {tool_name} not found", file=sys.stderr)
                            tool_grammar = None
                    except Exception as e:
                        if llama.verbose:
                            print(f"[DEBUG] Failed to create tool call grammar: {e}", file=sys.stderr)
                        tool_grammar = None

                    # Continue generation with tool grammar
                    tool_completion = llama.create_completion(
                        prompt=prompt + response_text,  # Include previous response
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        min_p=min_p,
                        typical_p=typical_p,
                        logprobs=top_logprobs if logprobs else None,
                        stream=stream,
                        stop=["</tool_call>"],  # Stop at tool call end tag
                        seed=seed,
                        max_tokens=max_tokens,
                        presence_penalty=presence_penalty,
                        frequency_penalty=frequency_penalty,
                        repeat_penalty=repeat_penalty,
                        tfs_z=tfs_z,
                        mirostat_mode=mirostat_mode,
                        mirostat_tau=mirostat_tau,
                        mirostat_eta=mirostat_eta,
                        model=model,
                        logits_processor=logits_processor,
                        stopping_criteria=stopping_criteria,
                        grammar=tool_grammar,  # Use tool call grammar
                        logit_bias=logit_bias,
                    )
                    
                    # Combine the completions
                    completion_or_chunks = {
                        **completion,
                        "choices": [{
                            **completion["choices"][0],
                            "text": response_text + tool_completion["choices"][0]["text"]
                        }]
                    }
                    
        else:
            # Regular completion for specific tool choice or no tools
            print("[DEBUG TOOLS] Starting initial completion...")
            completion_or_chunks = llama.create_completion(
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                typical_p=typical_p,
                logprobs=top_logprobs if logprobs else None,
                stream=stream,
                stop=[*stop, "<tool_call>"],  # Stop at tool call tag
            seed=seed,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            repeat_penalty=repeat_penalty,
            tfs_z=tfs_z,
            mirostat_mode=mirostat_mode,
            mirostat_tau=mirostat_tau,
            mirostat_eta=mirostat_eta,
            model=model,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            grammar=grammar,
            logit_bias=logit_bias,
        )
        # Flow normally until we hit a tool call
        if stream:
            # For streaming, we need to accumulate text until we see a tool call
            accumulated_text = ""
            completion_kwargs = {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "min_p": min_p,
                "typical_p": typical_p,
                "presence_penalty": presence_penalty,
                "frequency_penalty": frequency_penalty,
                "repeat_penalty": repeat_penalty,
                "tfs_z": tfs_z,
                "mirostat_mode": mirostat_mode,
                "mirostat_tau": mirostat_tau,
                "mirostat_eta": mirostat_eta,
                "model": model,
                "logits_processor": logits_processor,
                "logprobs": top_logprobs if logprobs else None,
                "max_tokens": max_tokens,
                "seed": seed,
            }
            
            for chunk in cast(Iterator[llama_types.CreateCompletionStreamResponse], completion_or_chunks):
                text = chunk["choices"][0]["text"]
                accumulated_text += text
                
                if accumulated_text.rstrip().endswith("<tool_call>"):
                    # Found tool call, switch to grammar mode
                    print("[DEBUG TOOLS] Found tool call, switching to grammar mode")
                    function_names = " | ".join([f'''"functions.{t["function"]["name"]}:"''' for t in tools]) if tools else ""
                    tool_call_gbnf = (
                        'root ::= "<tool_call>" "\\n" functions\n'
                        f"functions ::= {function_names}\n"
                    )
                    
                    # First get the tool name with grammar
                    try:
                        name_grammar = llama_grammar.LlamaGrammar.from_string(
                            tool_call_gbnf,
                            verbose=llama.verbose
                        )
                        
                        # Generate tool name
                        tool_name_completion = llama.create_completion(
                            prompt=prompt + accumulated_text,
                            temperature=0,
                            stream=False,
                            stop=[":"],
                            max_tokens=None,
                            grammar=name_grammar,
                        )
                        
                        # Get the selected tool
                        tool_name = tool_name_completion["choices"][0]["text"].split("\n")[-1][len("functions."):-1]
                        tool = next((t for t in tools if t["function"]["name"] == tool_name), None)
                        
                        if tool:
                            # Get tool parameters grammar
                            tool_grammar = _grammar_for_tool_parameters(tool, verbose=llama.verbose)
                            
                            # Continue generation with tool grammar
                            new_prompt = prompt + accumulated_text + tool_name_completion["choices"][0]["text"] + "\n"
                            for chunk in llama.create_completion(
                                prompt=new_prompt,
                                grammar=tool_grammar,
                                stream=True,
                                stop=["</tool_call>"],
                                **{k: v for k, v in completion_kwargs.items() if k != "stream" and k != "grammar"}
                            ):
                                yield from _convert_text_completion_chunks_to_chat(iter([chunk]))
                            
                            # After tool call, continue normal streaming
                            for chunk in llama.create_completion(
                                prompt=new_prompt,
                                stream=True,
                                **{k: v for k, v in completion_kwargs.items() if k != "stream"}
                            ):
                                yield from _convert_text_completion_chunks_to_chat(iter([chunk]))
                                
                    except Exception as e:
                        if llama.verbose:
                            print(f"[DEBUG] Failed to stream tool call: {e}", file=sys.stderr)
                        # Fall back to regular streaming without grammar
                        for chunk in llama.create_completion(
                            prompt=prompt + accumulated_text,
                            stream=True,
                            **{k: v for k, v in completion_kwargs.items() if k != "stream"}
                        ):
                            yield from _convert_text_completion_chunks_to_chat(iter([chunk]))
                else:
                    # Keep streaming normally until we find a tool call
                    yield from _convert_text_completion_chunks_to_chat(iter([chunk]))
        else:
            completion = cast(llama_types.CreateCompletionResponse, completion_or_chunks)
            response_text = completion["choices"][0]["text"]
            
            if response_text.rstrip().endswith("<tool_call>"):
                if llama.verbose:
                    print("[DEBUG] Found tool call tag, switching to tool grammar", file=sys.stderr)
                
                # First get the tool name using the tool name grammar
                try:
                    name_grammar = _grammar_for_tool_name(tools, verbose=llama.verbose)
                    
                    # Generate tool name
                    tool_name_completion = llama.create_completion(
                        prompt=prompt + response_text,
                        temperature=0,
                        stream=False,
                        stop=[":"],
                        max_tokens=None,
                        grammar=name_grammar,
                    )
                    
                    # Get the selected tool
                    tool_name = tool_name_completion["choices"][0]["text"].split("\n")[-1][len("functions."):-1]
                    tool = next((t for t in tools if t["function"]["name"] == tool_name), None)
                    
                    # Get the tool parameters grammar if tool exists
                    tool_grammar = _grammar_for_tool_parameters(tool, verbose=llama.verbose) if tool else None
                    
                    if tool is None and llama.verbose:
                        print(f"[DEBUG] Tool {tool_name} not found", file=sys.stderr)
                        
                except Exception as e:
                    if llama.verbose:
                        print(f"[DEBUG] Failed to create tool call grammar: {e}", file=sys.stderr)
                    tool_grammar = None

                # Continue generation with tool grammar
                tool_completion = llama.create_completion(
                    prompt=prompt + response_text,  # Include previous response
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                    typical_p=typical_p,
                    logprobs=top_logprobs if logprobs else None,
                    stream=stream,
                    stop=["</tool_call>"],  # Stop at tool call end tag
                    seed=seed,
                    max_tokens=max_tokens,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    repeat_penalty=repeat_penalty,
                    tfs_z=tfs_z,
                    mirostat_mode=mirostat_mode,
                    mirostat_tau=mirostat_tau,
                    mirostat_eta=mirostat_eta,
                    model=model,
                    logits_processor=logits_processor,
                    stopping_criteria=stopping_criteria,
                    grammar=tool_grammar,  # Use tool call grammar
                    logit_bias=logit_bias,
                )
                
                # Combine the completions
                completion_or_chunks = {
                    **completion,
                    "choices": [{
                        **completion["choices"][0],
                        "text": response_text + tool_completion["choices"][0]["text"]
                    }]
                }
            
        if tool is not None:
            tool_name = tool["function"]["name"]
            return _convert_completion_to_chat_function(
                tool_name, completion_or_chunks, stream
            )

        # Handle auto tool choice - parse response for function calls (like thinking tags)
        if tool_choice == "auto" and tools is not None and len(tools) > 0:
            # Only handle non-streaming for now (streaming tool calls need different approach)
            if not stream:
                completion_result = cast(llama_types.CreateCompletionResponse, completion_or_chunks)
                response_text = completion_result["choices"][0]["text"]

                if llama.verbose:
                    print(f"[DEBUG] Auto tool choice triggered. Response text: {response_text[:200]}...", file=sys.stderr)
                    print("[DEBUG] Looking for tool_call tags...", file=sys.stderr)

                # Parse the response similar to how template handles <think></think> and <tool_call></tool_call>
                message_content = response_text
                tool_call_json = None

                # Look for <tool_call> tags in the response
                tool_call_start_idx = response_text.find("<tool_call>")
                if tool_call_start_idx >= 0 and "</tool_call>" in response_text:
                    if llama.verbose:
                        print("[DEBUG] Found tool_call tags, attempting to parse", file=sys.stderr)
                    try:
                        # Extract content between <tool_call> tags (like template does)
                        tool_call_start = tool_call_start_idx + len("<tool_call>")
                        tool_call_end = response_text.find("</tool_call>")
                        
                        # Switch to grammar strict mode for tool call content
                        if tool_call_start_idx > 0:
                            # Get content before tool call tag
                            pre_tool_call = response_text[:tool_call_start_idx].strip()
                            if pre_tool_call:
                                # Store pre-tool call content
                                message_content = pre_tool_call

                        if tool_call_start >= 0 and tool_call_end > tool_call_start:
                            # Switch to grammar strict mode for tool call content
                            tool_call_content = response_text[tool_call_start:tool_call_end].strip()
                            if llama.verbose:
                                print(f"[DEBUG] Extracted tool_call content: {tool_call_content}", file=sys.stderr)
                                print("[DEBUG] Switching to grammar strict mode for tool call", file=sys.stderr)
                            
                            # Create JSON grammar for tool calls
                            tool_call_schema = {
                                "type": "object",
                                "required": ["type", "name", "input"],
                                "properties": {
                                    "type": {"type": "string", "enum": ["tool_use"]},
                                    "name": {"type": "string"},
                                    "input": {"type": "object"}
                                }
                            }
                            
                            # Create grammar for tool call JSON
                            try:
                                grammar = llama_grammar.LlamaGrammar.from_json_schema(
                                    json.dumps(tool_call_schema), 
                                    verbose=llama.verbose
                                )
                            except Exception as e:
                                if llama.verbose:
                                    print(f"[DEBUG] Failed to create tool call grammar: {e}", file=sys.stderr)
                                grammar = None
                                
                            parsed_json = json.loads(tool_call_content)

                            # Check if it's a valid tool call
                            if (parsed_json.get("type") == "tool_use" and
                                "name" in parsed_json and
                                "input" in parsed_json):

                                if llama.verbose:
                                    print(f"[DEBUG] Valid tool call found: {parsed_json.get('name')}", file=sys.stderr)
                                tool_call_json = parsed_json
                                # Extract message content before the <tool_call> tag
                                message_content = response_text[:response_text.find("<tool_call>")].strip()

                    except json.JSONDecodeError as e:
                        # Not valid JSON, treat as pure message
                        if llama.verbose:
                            print(f"[DEBUG] Tool call JSON parsing failed: {e}. JSON text: {tool_call_content[:200]}...", file=sys.stderr)
                        pass
                else:
                    if llama.verbose:
                        has_start = "<tool_call>" in response_text
                        has_end = "</tool_call>" in response_text
                        print(f"[DEBUG] Tool call tags not found. Has start tag: {has_start}, Has end tag: {has_end}", file=sys.stderr)

                if llama.verbose:
                    print(f"[DEBUG] Final tool_call_json: {tool_call_json is not None}", file=sys.stderr)

                # If we found a valid tool call, build the response
                if tool_call_json:
                    if llama.verbose:
                        print(f"[DEBUG] Building tool call response for {tool_call_json['name']}", file=sys.stderr)
                    tool_name = tool_call_json["name"]
                    tool_input = tool_call_json["input"]
                    tool_id = "call_0_" + tool_name + "_" + completion_result["id"]

                    return {
                        "id": "chat" + completion_result["id"],
                        "object": "chat.completion",
                        "created": completion_result["created"],
                        "model": completion_result["model"],
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": message_content if message_content else None,
                                "tool_calls": [{
                                    "id": tool_id,
                                    "type": "function",
                                    "function": {
                                        "name": tool_name,
                                        "arguments": json.dumps(tool_input),
                                    },
                                }],
                            },
                            "logprobs": _convert_text_completion_logprobs_to_chat(
                                completion_result["choices"][0]["logprobs"]
                            ),
                            "finish_reason": "tool_calls",
                        }],
                        "usage": completion_result["usage"],
                    }

        return _convert_completion_to_chat(completion_or_chunks, stream=stream)

    return chat_completion_handler


def _stream_tool_calls(
    llama: llama.Llama,
    prompt: str,
    tools: List[llama_types.ChatCompletionTool],
    tool_name: str,
    completion_kwargs: dict[str, Any],
    follow_up_gbnf_tool_grammar: str,
) -> Iterator[llama_types.ChatCompletionChunk]:
    # Generate a tool call completions
    tool = next((tool for tool in tools if tool["function"]["name"] == tool_name), None)
    completions: List[llama_types.CreateCompletionResponse] = []
    completions_tool_name: List[str] = []
    finish_reason_chat_chunk = None
    while tool is not None and len(completions) <= 16:
        # Generate the parameter values for the selected tool
        prompt += f"functions.{tool_name}:\n"
        try:
            grammar = llama_grammar.LlamaGrammar.from_json_schema(
                json.dumps(tool["function"]["parameters"]), verbose=llama.verbose
            )
        except Exception as e:
            warnings.warn(
                f"Failed to parse function body as JSON schema, falling back to default grammar\n\n{e}",
                category=RuntimeWarning,
                stacklevel=2,
            )
            grammar = llama_grammar.LlamaGrammar.from_string(
                llama_grammar.JSON_GBNF, verbose=llama.verbose
            )
        completion_or_chunks = llama.create_completion(
            prompt=prompt,
            **{
                **completion_kwargs,
                "max_tokens": None,
                "grammar": grammar,
            },
        )
        chunks: List[llama_types.CreateCompletionResponse] = []
        chat_chunks = _convert_completion_to_chat_function(
            tool_name,
            _accumulate_chunks(completion_or_chunks, chunks),  # type: ignore[arg-type]
            stream=True,
        )
        for chat_chunk in chat_chunks:
            # Don't return the finish_reason chunk
            if chat_chunk["choices"] and chat_chunk["choices"][0].get("finish_reason"):
                finish_reason_chat_chunk = chat_chunk
                break
            # Update this tool call's index
            if chat_chunk["choices"] and chat_chunk["choices"][0]["delta"].get("tool_calls"):
                chat_chunk["choices"][0]["delta"]["tool_calls"][0]["index"] = len(completions)
            yield chat_chunk
        completion = _convert_chunks_to_completion(chunks)
        completions.append(completion)
        completions_tool_name.append(tool_name)
        prompt += completion["choices"][0]["text"]
        prompt += "\n"
        # Determine whether to call another tool or stop
        response = cast(
            llama_types.CreateCompletionResponse,
            llama.create_completion(
                prompt=prompt,
                **{
                    **completion_kwargs,
                    "temperature": 0,
                    "stream": False,
                    "stop": [*completion_kwargs["stop"], ":", "</function_calls>"],
                    "max_tokens": None,
                    "grammar": llama_grammar.LlamaGrammar.from_string(
                        follow_up_gbnf_tool_grammar, verbose=llama.verbose
                    ),
                },
            ),
        )
        tool_name = response["choices"][0]["text"][len("functions.") :]
        tool = next((tool for tool in tools if tool["function"]["name"] == tool_name), None)
    # Yield the finish_reason chunk
    if finish_reason_chat_chunk is not None:
        yield finish_reason_chat_chunk

def _accumulate_chunks(
    chunks_iterator: Iterator[llama_types.CreateCompletionStreamResponse],
    chunks_list: List[llama_types.CreateCompletionStreamResponse],
) -> Iterator[llama_types.CreateCompletionStreamResponse]:
    for chunk in chunks_iterator:
        chunks_list.append(chunk)
        yield chunk
        

def _convert_chunks_to_completion(
    chunks: List[llama_types.CreateCompletionStreamResponse],
) -> llama_types.CreateCompletionResponse:
    """Convert a list of completion chunks to a completion."""
    # Accumulate completion response values
    text: str = ""
    finish_reason: Optional[str] = None
    logprobs: Optional[llama_types.CompletionLogprobs] = None
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    completion_id: Optional[str] = None
    completion_model: Optional[str] = None
    completion_created: Optional[int] = None
    for chunk in chunks:
        # Extract the id, model, and created values from the first chunk
        if completion_id is None:
            completion_id = chunk["id"]
            completion_model = chunk["model"]
            completion_created = chunk["created"]
        # Extract the usage if present in the chunk
        usage = chunk.get("usage")
        if usage:
            prompt_tokens += usage.get("prompt_tokens", 0)
            completion_tokens += usage.get("completion_tokens", 0)
            total_tokens += usage.get("total_tokens", 0)
        # Accumulate the chunk text
        choice = chunk["choices"][0]
        text += choice.get("text", "")
        # Extract the finish_reason and logprobs if present in the chunk
        if choice.get("finish_reason"):
            finish_reason = choice["finish_reason"]
        if choice.get("logprobs"):
            logprobs = choice["logprobs"]
    # Create the completion response
    completion: llama_types.CreateCompletionResponse = {
        "id": completion_id or "unknown_id",
        "object": "text_completion",
        "created": completion_created or 0,
        "model": completion_model or "unknown_model",
        "choices": [
            {
                "text": text,
                "index": 0,
                "logprobs": logprobs,  # TODO: Improve accumulation of logprobs
                "finish_reason": finish_reason,  # type: ignore[typeddict-item]
            }
        ],
    }
    # Add usage section if present in the chunks
    if (prompt_tokens + completion_tokens + total_tokens) > 0:
        completion["usage"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
    return completion


def _convert_text_completion_chunks_to_chat(
    chunks: Iterator[llama_types.CreateCompletionStreamResponse],
) -> Iterator[llama_types.ChatCompletionChunk]:
    for i, chunk in enumerate(chunks):
        if i == 0:
            yield {
                "id": "chat" + chunk["id"],
                "model": chunk["model"],
                "created": chunk["created"],
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                        },
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
            }
        yield {
            "id": "chat" + chunk["id"],
            "model": chunk["model"],
            "created": chunk["created"],
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": 0,
                    "delta": (
                        {
                            "content": chunk["choices"][0]["text"],
                        }
                        if chunk["choices"][0]["finish_reason"] is None
                        else {}
                    ),
                    "logprobs": _convert_text_completion_logprobs_to_chat(
                        chunk["choices"][0]["logprobs"]
                    ),
                    "finish_reason": chunk["choices"][0]["finish_reason"],
                }
            ],
            **(
                {"usage": chunk["usage"]}
                if "usage" in chunk and chunk["usage"] is not None
                else {}
            ),
        }


def _convert_text_completion_logprobs_to_chat(
    logprobs: Optional[llama_types.CompletionLogprobs],
) -> llama_types.ChatCompletionLogprobs:
    if logprobs is None:
        return None

    return {
        "content": [
            {
                "token": token,
                "bytes": None,
                "logprob": logprob,
                "top_logprobs": [
                    {
                        "token": top_token,
                        "logprob": top_logprob,
                        "bytes": None,
                    }
                    for top_token, top_logprob in top_logprobs.items()
                ],
            }
            for (token, logprob, top_logprobs) in zip(
                logprobs["tokens"], logprobs["token_logprobs"], logprobs["top_logprobs"]
            )
        ],
        "refusal": None,
    }


def _convert_text_completion_to_chat(
    completion: llama_types.Completion,
) -> llama_types.ChatCompletion:
    assert "usage" in completion
    return {
        "id": "chat" + completion["id"],
        "object": "chat.completion",
        "created": completion["created"],
        "model": completion["model"],
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": completion["choices"][0]["text"],
                },
                "logprobs": _convert_text_completion_logprobs_to_chat(
                    completion["choices"][0]["logprobs"]
                ),
                "finish_reason": completion["choices"][0]["finish_reason"],
            }
        ],
        "usage": completion["usage"],
    }


def _convert_completion_to_chat(
    completion_or_chunks: Union[
        llama_types.CreateCompletionResponse,
        Iterator[llama_types.CreateCompletionStreamResponse],
    ],
    stream: bool = False,
) -> Union[
    llama_types.CreateChatCompletionResponse, Iterator[llama_types.ChatCompletionChunk]
]:
    if stream:
        chunks: Iterator[llama_types.CreateCompletionStreamResponse] = (
            completion_or_chunks  # type: ignore
        )
        return _convert_text_completion_chunks_to_chat(chunks)
    else:
        completion: llama_types.Completion = completion_or_chunks  # type: ignore
        return _convert_text_completion_to_chat(completion)


def _convert_completion_to_chat_function(
    tool_name: str,
    completion_or_chunks: Union[
        llama_types.CreateCompletionResponse,
        Iterator[llama_types.CreateCompletionStreamResponse],
    ],
    stream: bool,
):
    if not stream:
        completion: llama_types.CreateCompletionResponse = completion_or_chunks  # type: ignore
        assert "usage" in completion
        tool_id = "call_" + "_0_" + tool_name + "_" + completion["id"]
        # TODO: Fix for legacy function calls
        chat_completion: llama_types.CreateChatCompletionResponse = {
            "id": "chat" + completion["id"],
            "object": "chat.completion",
            "created": completion["created"],
            "model": completion["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "function_call": {
                            "name": tool_name,
                            "arguments": completion["choices"][0]["text"],
                        },
                        "tool_calls": [
                            {
                                "id": tool_id,
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": completion["choices"][0]["text"],
                                },
                            }
                        ],
                    },
                    "logprobs": _convert_text_completion_logprobs_to_chat(
                        completion["choices"][0]["logprobs"]
                    ),
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": completion["usage"],
        }
        return chat_completion
    else:
        chunks: Iterator[llama_types.CreateCompletionStreamResponse] = (
            completion_or_chunks  # type: ignore
        )

        def _stream_response_to_function_stream(
            chunks: Iterator[llama_types.CreateCompletionStreamResponse],
        ) -> Iterator[llama_types.CreateChatCompletionStreamResponse]:
            # blank first message
            first = True
            id_ = None
            created = None
            model = None
            tool_id = None
            for chunk in chunks:
                if first:
                    id_ = "chat" + chunk["id"]
                    created = chunk["created"]
                    model = chunk["model"]
                    tool_id = "call_" + "_0_" + tool_name + "_" + chunk["id"]
                    response = {
                        "id": id_,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "finish_reason": None,
                                "logprobs": None,
                                "delta": {
                                    "role": "assistant",
                                    "content": None,
                                    "function_call": None,
                                    "tool_calls": None,
                                },
                            }
                        ],
                    }
                    if "usage" in chunk:
                        response["usage"] = chunk["usage"]
                    yield response

                    response = {
                        "id": "chat" + chunk["id"],
                        "object": "chat.completion.chunk",
                        "created": chunk["created"],
                        "model": chunk["model"],
                        "choices": [
                            {
                                "index": 0,
                                "finish_reason": None,
                                "logprobs": _convert_text_completion_logprobs_to_chat(
                                    chunk["choices"][0]["logprobs"]
                                ),
                                "delta": {
                                    "role": None,
                                    "content": None,
                                    "function_call": {
                                        "name": tool_name,
                                        "arguments": chunk["choices"][0]["text"],
                                    },
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": tool_id,
                                            "type": "function",
                                            "function": {
                                                "name": tool_name,
                                                "arguments": chunk["choices"][0][
                                                    "text"
                                                ],
                                            },
                                        }
                                    ],
                                },
                            }
                        ],
                    }
                    if "usage" in chunk:
                        response["usage"] = chunk["usage"]
                    yield response
                    first = False
                    continue

                assert tool_id is not None
                response = {
                    "id": "chat" + chunk["id"],
                    "object": "chat.completion.chunk",
                    "created": chunk["created"],
                    "model": chunk["model"],
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": None,
                            "logprobs": _convert_text_completion_logprobs_to_chat(
                                chunk["choices"][0]["logprobs"]
                            ),
                            "delta": {
                                "role": None,
                                "content": None,
                                "function_call": {
                                    "name": tool_name,
                                    "arguments": chunk["choices"][0]["text"],
                                },
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": tool_id,
                                        "type": "function",
                                        "function": {
                                            "name": tool_name,
                                            "arguments": chunk["choices"][0]["text"],
                                        },
                                    }
                                ],
                            },
                        }
                    ],
                }
                if "usage" in chunk:
                    response["usage"] = chunk["usage"]
                yield response

            if id_ is not None and created is not None and model is not None:
                response = {
                    "id": id_,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "tool_calls",
                            "logprobs": None,
                            "delta": {
                                "role": None,
                                "content": None,
                                "function_call": None,
                                "tool_calls": None,
                            },
                        }
                    ],
                }
                if "usage" in chunk:
                    response["usage"] = chunk["usage"]
                yield response

        return _stream_response_to_function_stream(chunks)


def _grammar_for_json(verbose: bool = False):
    return llama_grammar.LlamaGrammar.from_string(
        llama_grammar.JSON_GBNF, verbose=verbose
    )


def _grammar_for_json_schema(
    schema: str, verbose: bool = False, fallback_to_json: bool = True
):
    try:
        return llama_grammar.LlamaGrammar.from_json_schema(schema, verbose=verbose)
    except Exception as e:
        if fallback_to_json:
            return _grammar_for_json(verbose=verbose)
        else:
            raise e


def _grammar_for_response_format(
    response_format: llama_types.ChatCompletionRequestResponseFormat,
    verbose: bool = False,
):
    if response_format["type"] != "json_object":
        return None

    if "schema" in response_format:
        return _grammar_for_json_schema(
            json.dumps(response_format["schema"]), verbose=verbose
        )
    else:
        return _grammar_for_json(verbose=verbose)


def _grammar_for_tool_name(
    tools: List[llama_types.ChatCompletionTool],
    verbose: bool = False,
) -> llama.LlamaGrammar:
    """Create a grammar that enforces the tool name format.
    
    The grammar ensures the response starts with <tool_call> followed by a valid tool name."""
    function_names = " | ".join([f'''"functions.{t["function"]["name"]}:"''' for t in tools])
    tool_call_gbnf = (
        'root ::= "<tool_call>" "\\n" functions\n'
        f"functions ::= {function_names}\n"
    )
    return llama_grammar.LlamaGrammar.from_string(tool_call_gbnf, verbose=verbose)


def _grammar_for_tool_parameters(
    tool: llama_types.ChatCompletionTool,
    verbose: bool = False,
) -> llama.LlamaGrammar:
    """Create a grammar that enforces the tool parameters format based on the tool's JSON schema.
    
    Falls back to generic JSON grammar if schema parsing fails."""
    try:
        return llama_grammar.LlamaGrammar.from_json_schema(
            json.dumps(tool["function"]["parameters"]),
            verbose=verbose
        )
    except Exception as e:
        if verbose:
            print(f"[DEBUG] Failed to parse function parameters as JSON schema: {e}", file=sys.stderr)
        return llama_grammar.LlamaGrammar.from_string(
            llama_grammar.JSON_GBNF,
            verbose=verbose
        )
