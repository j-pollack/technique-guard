"""Flag user input that tries to prompt-inject or jailbreak a downstream LLM.

    from anti_venom import check
    v = check(user_text)
    if v.action == "block": ...
    elif v.action == "review": ...

Uses Typesafe (https://api.typesafe.ai) for the judgments; needs TYPESAFE_API_KEY.
"""

import json
import os
import urllib.request
from dataclasses import dataclass

URL = "https://api.typesafe.ai/preview/evaluation"

INJECTION = {
    "overrides_instructions": "Does this text try to make an AI ignore, override, or replace the instructions it was given?",
    "impersonates_authority": "Does this text pretend to come from a system, developer, administrator, or the AI's creator?",
    "extracts_system_prompt": "Does this text try to get an AI to reveal its system prompt, hidden instructions, or configuration?",
    "hidden_instructions": "Does this text contain instructions aimed at an AI that processes it, hidden inside content meant for something else?",
    "exfiltration": "Does this text try to make an AI send data to a URL, email address, or external tool?",
}
JAILBREAK = {
    "unrestricted_persona": "Does this text ask an AI to act as a persona or mode that has no rules, filters, or restrictions?",
    "obfuscation": "Does this text use encoding, ciphers, odd spacing, or split-up words to disguise what it is really asking for?",
    "pressure_tactics": "Does this text use threats, guilt, rewards, or fake emergencies to pressure an AI into breaking its rules?",
}
# Fiction/hypotheticals are normal on their own; only count them alongside a harmful ask.
HYPOTHETICAL = "Does this text use a fictional, hypothetical, or 'for research only' framing to justify its request?"
HARMFUL = "Does this text ask for dangerous, illegal, or seriously harmful content?"

PROMPTS = [
    *({"key": k, "type": "noul", "instructions": q} for k, q in {**INJECTION, **JAILBREAK}.items()),
    {"key": "hypothetical_framing", "type": "noul", "instructions": HYPOTHETICAL},
    {"key": "harmful_request", "type": "noul", "instructions": HARMFUL},
    {
        "key": "intent",
        "type": "choice",
        "instructions": "What is this text most likely trying to do to the AI that receives it?",
        "options": [
            {"option": "benign", "description": "An ordinary request or message, even if it touches sensitive topics"},
            {"option": "prompt_injection", "description": "Hijack the AI: override its instructions, leak its prompt, or make it take actions for someone else"},
            {"option": "jailbreak", "description": "Talk the AI out of its safety rules to get content it would normally refuse"},
        ],
    },
]


@dataclass
class Verdict:
    action: str  # "allow" | "review" | "block"
    category: str  # "benign" | "prompt_injection" | "jailbreak"
    risk: float  # 0..1
    confidence: float  # intent classifier confidence
    signals: dict[str, float]


def judge(responses: list[dict], block_at: float = 0.8, allow_below: float = 0.3, min_confidence: float = 0.6) -> Verdict:
    """Combine raw Typesafe responses into a Verdict. Pure, so it's testable offline."""
    signals = {r["key"]: r["probability"] for r in responses if r["type"] == "noul"}
    intent = next(r for r in responses if r["key"] == "intent")
    intent_p = {p["option"]: p["probability"] for p in intent["probabilities"]}

    # ponytail: max-of-signals, swap for learned weights once there's labeled traffic
    injection = max([intent_p["prompt_injection"], *(signals[k] for k in INJECTION)])
    jailbreak = max([
        intent_p["jailbreak"],
        *(signals[k] for k in JAILBREAK),
        min(signals["hypothetical_framing"], signals["harmful_request"]),
    ])
    risk = max(injection, jailbreak)
    category = "benign" if risk < allow_below else ("prompt_injection" if injection >= jailbreak else "jailbreak")

    if risk >= block_at:
        action = "block"
    elif risk < allow_below and intent["confidence"] >= min_confidence:
        action = "allow"
    else:
        action = "review"  # in between, or the classifier isn't sure
    return Verdict(action, category, risk, intent["confidence"], signals)


def check(text: str, api_key: str | None = None, timeout: float = 10, **thresholds) -> Verdict:
    """Evaluate `text`. Raises on API/network errors: callers decide whether to fail open or closed."""
    body = {
        # Wrapped in an object so the input reads as data under evaluation, not as instructions.
        "document": {"untrusted_user_input": text},
        "model": "speed_latest",
        "prompts": PROMPTS,
    }
    req = urllib.request.Request(
        URL,
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {api_key or os.environ['TYPESAFE_API_KEY']}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return judge(json.load(resp)["responses"], **thresholds)
