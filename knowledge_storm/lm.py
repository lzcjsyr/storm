# File role: Language-model adapter layer with unified OpenAI-compatible invocation behavior.
# Relation: Engines consume LitellmModel via LM config objects and lm_routing TOML.
import functools
import os
import threading
from pathlib import Path
from typing import Literal, Optional

import ujson


############################
# Core logic adapted from dspy's LM client behavior.

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    if "LITELLM_LOCAL_MODEL_COST_MAP" not in os.environ:
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    import litellm

    litellm.drop_params = True
    litellm.telemetry = False

from litellm.caching.caching import Cache


disk_cache_dir = os.path.join(Path.home(), ".storm_local_cache")
litellm.cache = Cache(disk_cache_dir=disk_cache_dir, type="disk")

LM_LRU_CACHE_MAX_SIZE = 3000


class LM:
    def __init__(
        self,
        model,
        model_type="chat",
        temperature=0.0,
        max_tokens=1000,
        cache=True,
        **kwargs,
    ):
        self.model = model
        self.model_type = model_type
        self.cache = cache
        self.kwargs = dict(temperature=temperature, max_tokens=max_tokens, **kwargs)
        self.history = []

        if "o1-" in model:
            assert (
                max_tokens >= 5000 and temperature == 1.0
            ), "OpenAI's o1-* models require passing temperature=1.0 and max_tokens >= 5000 to `dspy.LM(...)`"

    def __call__(self, prompt=None, messages=None, **kwargs):
        cache = kwargs.pop("cache", self.cache)
        messages = messages or [{"role": "user", "content": prompt}]
        kwargs = {**self.kwargs, **kwargs}

        if self.model_type == "chat":
            completion = cached_litellm_completion if cache else litellm_completion
        else:
            completion = (
                cached_litellm_text_completion if cache else litellm_text_completion
            )

        response = completion(
            ujson.dumps(dict(model=self.model, messages=messages, **kwargs))
        )
        outputs = [
            c.message.content if hasattr(c, "message") else c["text"]
            for c in response["choices"]
        ]

        safe_kwargs = {k: v for k, v in kwargs.items() if not k.startswith("api_")}
        entry = dict(
            prompt=prompt,
            messages=messages,
            kwargs=safe_kwargs,
            response=response,
            outputs=outputs,
            usage=dict(response["usage"]),
            cost=response.get("_hidden_params", {}).get("response_cost"),
        )
        self.history.append(entry)

        return outputs

    def inspect_history(self, n: int = 1):
        _inspect_history(self, n)


@functools.lru_cache(maxsize=LM_LRU_CACHE_MAX_SIZE)
def cached_litellm_completion(request):
    return litellm_completion(request, cache={"no-cache": False, "no-store": False})


def litellm_completion(request, cache={"no-cache": True, "no-store": True}):
    kwargs = ujson.loads(request)
    return litellm.completion(cache=cache, **kwargs)


@functools.lru_cache(maxsize=LM_LRU_CACHE_MAX_SIZE)
def cached_litellm_text_completion(request):
    return litellm_text_completion(
        request, cache={"no-cache": False, "no-store": False}
    )


def litellm_text_completion(request, cache={"no-cache": True, "no-store": True}):
    kwargs = ujson.loads(request)

    model = kwargs.pop("model").split("/", 1)
    provider, model = model[0] if len(model) > 1 else "openai", model[-1]

    api_key = kwargs.pop("api_key", None) or os.getenv(f"{provider}_API_KEY")
    api_base = kwargs.pop("api_base", None) or os.getenv(f"{provider}_API_BASE")

    prompt = "\n\n".join(
        [x["content"] for x in kwargs.pop("messages")] + ["BEGIN RESPONSE:"]
    )

    return litellm.text_completion(
        cache=cache,
        model=f"text-completion-openai/{model}",
        api_key=api_key,
        api_base=api_base,
        prompt=prompt,
        **kwargs,
    )


def _green(text: str, end: str = "\n"):
    return "\x1b[32m" + str(text).lstrip() + "\x1b[0m" + end


def _red(text: str, end: str = "\n"):
    return "\x1b[31m" + str(text) + "\x1b[0m" + end


def _inspect_history(lm, n: int = 1):
    for item in lm.history[-n:]:
        messages = item["messages"] or [{"role": "user", "content": item["prompt"]}]
        outputs = item["outputs"]

        print("\n\n\n")
        for msg in messages:
            print(_red(f"{msg['role'].capitalize()} message:"))
            print(msg["content"].strip())
            print("\n")

        print(_red("Response:"))
        print(_green(outputs[0].strip()))

        if len(outputs) > 1:
            choices_text = f" \t (and {len(outputs)-1} other completions)"
            print(_red(choices_text, end=""))

    print("\n\n\n")


############################


class LitellmModel(LM):
    """A lightweight wrapper over litellm for OpenAI-compatible and provider-native models."""

    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        api_key: Optional[str] = None,
        model_type: Literal["chat", "text"] = "chat",
        **kwargs,
    ):
        super().__init__(model=model, api_key=api_key, model_type=model_type, **kwargs)
        self._token_usage_lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def log_usage(self, response):
        usage_data = response.get("usage")
        if usage_data:
            with self._token_usage_lock:
                self.prompt_tokens += usage_data.get("prompt_tokens", 0)
                self.completion_tokens += usage_data.get("completion_tokens", 0)

    def get_usage_and_reset(self):
        usage = {
            self.model
            or self.kwargs.get("model")
            or self.kwargs.get("engine"): {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
            }
        }
        self.prompt_tokens = 0
        self.completion_tokens = 0

        return usage

    def __call__(self, prompt=None, messages=None, **kwargs):
        cache = kwargs.pop("cache", self.cache)
        messages = messages or [{"role": "user", "content": prompt}]
        kwargs = {**self.kwargs, **kwargs}

        if self.model_type == "chat":
            completion = cached_litellm_completion if cache else litellm_completion
        else:
            completion = (
                cached_litellm_text_completion if cache else litellm_text_completion
            )

        response = completion(
            ujson.dumps(dict(model=self.model, messages=messages, **kwargs))
        )
        response_dict = response.json()
        self.log_usage(response_dict)
        outputs = [
            c.message.content if hasattr(c, "message") else c["text"]
            for c in response["choices"]
        ]

        safe_kwargs = {k: v for k, v in kwargs.items() if not k.startswith("api_")}
        entry = dict(
            prompt=prompt,
            messages=messages,
            kwargs=safe_kwargs,
            response=response_dict,
            outputs=outputs,
            usage=dict(response_dict["usage"]),
            cost=response.get("_hidden_params", {}).get("response_cost"),
        )
        self.history.append(entry)

        return outputs


__all__ = ["LM", "LitellmModel"]
