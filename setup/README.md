# Hack Armenia — LLM access

You get an **OpenRouter** API key. OpenRouter is an OpenAI-compatible gateway that forwards
your calls to **DeepSeek** using the organisers' account, so you need no DeepSeek account and
pay nothing.

```
you  →  OpenRouter  →  DeepSeek
```

| | |
|---|---|
| Endpoint | `https://openrouter.ai/api/v1` |
| Model | `deepseek/deepseek-v4-pro` |
| Budget | $120 per key, one-time (no reset) |
| Expires | 7 days from issue |

> **Treat the key like a password.** Don't commit it, don't paste it in chat, don't share it.
> Each key is tracked separately, so usage is attributable to you.

## Setup

```bash
pip install openai
export OPENROUTER_API_KEY="sk-or-..."      # your key
```

To keep it across sessions, put that `export` line in `~/.zshrc` (or `~/.bashrc`) and
`source` it. Never hard-code the key in a file you commit.

## Run

```bash
python example.py
python example.py "Summarise this idea in two sentences: ..."
```

Expected output:

```
Hello!

[tokens] prompt=10 completion=35 total=45
```

Token counts go to stderr, so `python example.py > out.txt` keeps only the reply.

## The code

Any OpenAI client works — just point `base_url` at OpenRouter:

```python
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)

response = client.chat.completions.create(
    model="deepseek/deepseek-v4-pro",
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
    max_tokens=256,
)
print(response.choices[0].message.content)
```

Streaming, `temperature`, multi-turn `messages`, JSON mode and tool calls all work as they do
against the OpenAI API.

## Check your remaining budget

```bash
curl -s https://openrouter.ai/api/v1/key \
  -H "Authorization: Bearer $OPENROUTER_API_KEY" | jq .data
```

Look at `limit` (your $120), `usage` (spent) and `limit_remaining` (left). No `jq`? Pipe to
`python3 -m json.tool` instead.

Check `limit_remaining` after your first real session to get a feel for the burn rate. Agent
loops resend the whole conversation every turn, so they cost far more than the same number of
one-off questions.

## Troubleshooting

| Error | Meaning | Fix |
|---|---|---|
| `401` | Bad or truncated key | Re-copy the key; check for stray whitespace |
| `402` | Budget exhausted or key expired | Request a new key |
| `404` / model not found | Wrong model slug | Use `deepseek/deepseek-v4-pro` |
| `OPENROUTER_API_KEY is not set` | Env var missing in this shell | Re-run the `export` |

## Using it from Claude Code

```bash
export OPENROUTER_API_KEY="sk-or-..."
export ANTHROPIC_BASE_URL="https://openrouter.ai/api"   # note: /api, NOT /api/v1
export ANTHROPIC_API_KEY="$OPENROUTER_API_KEY"
export ANTHROPIC_AUTH_TOKEN=""                          # must be explicitly empty
export CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1
export ANTHROPIC_MODEL=deepseek-v4-pro[1m]
export ANTHROPIC_DEFAULT_OPUS_MODEL=deepseek-v4-pro[1m]
export ANTHROPIC_DEFAULT_SONNET_MODEL=deepseek-v4-pro[1m]
```

Two things trip people up: the base URL ends in `/api` (adding `/v1` fails), and
`ANTHROPIC_AUTH_TOKEN` must be set to empty or Claude Code tries to authenticate against
Anthropic instead of the gateway.

Claude Code is tuned for Anthropic's models — it runs on DeepSeek, but multi-step tool use is
less reliable. It's a cost trade, not a like-for-like swap.
