import re

from inference import generate

AGGREGATE_INSTRUCTION = """[Instruction]
You are an impartial aggregator. Given a user prompt and multiple candidate answers, produce one final answer.

Your aggregation is merge-leaning:
- Primarily combine, edit, and reorganize what is already in the candidates.
- Do not add new substantive content or external facts.

Rules:
1) Follow the user's requested format, constraints, and style exactly.
2) Use only information explicitly stated in the candidates. You may add minimal connective wording for readability.
3) If candidates disagree and you cannot resolve it from the candidates, omit the claim or mark it as uncertain.
4) Remove redundancy and irrelevant parts; choose the clearest phrasing among candidates.
5) If JSON/code/strict format is required, keep it valid and do not introduce new APIs/libraries not present in candidates.
6) Two-turn consistency: if the prompt is multi-turn, treat the turn-1 aggregated answer as the assistant's previous message when producing the turn-2 aggregated answer.

After synthesizing, output ONLY the final aggregated answer, with no extra commentary.
"""

JUDGE_WO_REF_INSTRUCTION = (
    '[Instruction]\n'
    'You are an impartial judge evaluating the quality of an assistant\'s response.'
    'Evaluate the response based on the following criteria:'
    '- helpfulness'
    '- relevance'
    '- accuracy'
    '- clarity'
    '- completeness'

    'Think carefully about the response before rating it.'
    'Provide at most three short sentences summarizing the main reason for your score.'
    'Then, on a new final line, output the rating strictly in the format "[[rating]]",'
    'where rating is an integer from 1 to 10, for example: "[[5]]".'
    'Do not use any other rating format.'
    'Do not output anything after the final rating line.'
)

# JUDGE_WO_REF_INSTRUCTION = (
#     '[Instruction]\n'
#     'Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant '
#     'to the user question displayed below. Your evaluation should consider factors such as the helpfulness, '
#     'relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by '
#     'providing a short explanation. Be as objective as possible. After providing your explanation, you must '
#     'rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: '
#     '"Rating: [[5]]".'
# )

JUDGE_WITH_REF_INSTRUCTION = (
    '[Instruction]\n'
    'You are an impartial judge evaluating the quality of an assistant\'s response using a reference answer.'
    'Evaluate the response by comparing it with the reference answer based on the following criteria:'
    '- correctness'
    '- helpfulness'
    '- relevance'
    '- clarity'
    '- completeness'

    'Think carefully about the response before rating it.'
    'If the assistant\'s response misses important points or contains mistakes compared with the reference answer, take that into account.'
    'Provide at most three short sentences summarizing the main reason for your score.'
    'Then, on a new final line, output the rating strictly in the format "[[rating]]",'
    'where rating is an integer from 1 to 10, for example: "[[5]]".'
    'Do not use any other rating format.'
    'Do not output anything after the final rating line.'
)

# JUDGE_WITH_REF_INSTRUCTION = (
#     '[Instruction]\n'
#     "Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant "
#     "to the user question displayed below. Your evaluation should consider correctness and helpfulness. "
#     "You will be given a reference answer and the assistant's answer. Begin your evaluation by comparing "
#     "the assistant's answer with the reference answer. Identify and correct any mistakes. Be as objective "
#     'as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by '
#     'strictly following this format: "[[rating]]", for example: "Rating: [[5]]".'
# )


def build_aggregate_prompt(hist, y, rng, turn):
    """Return the aggregate prompt string only (for use with generate_batch)."""
    if turn == 1:
        x1 = hist
        y1 = list(y)
        rng.shuffle(y1)
        y1_block = "\n\n".join(
            f"[The Start of Candidate {i+1} Answer]\n{ans}\n[The End of Candidate {i+1} Answer]"
            for i, ans in enumerate(y1)
        )
        return f"""{AGGREGATE_INSTRUCTION}

[Question]
{x1}

<|The Start of Candidates|>
{y1_block}
<|The End of Candidates|>
"""
    x1, y1S, x2 = hist
    y2 = list(y)
    rng.shuffle(y2)
    y2_block = "\n\n".join(
        f"[The Start of Candidate {i+1} Answer]\n{ans}\n[The End of Candidate {i+1} Answer]"
        for i, ans in enumerate(y2)
    )
    return f"""{AGGREGATE_INSTRUCTION}

<|The Start of Previous Conversation with User|>
### User:
{x1}

### Assistant:
{y1S}

### User:
{x2}
<|The End of Previous Conversation with User|>

<|The Start of Candidates|>
{y2_block}
<|The End of Candidates|>
"""


def aggregate(hist, y, rng, turn, aggregate_config):
    prompt = build_aggregate_prompt(hist, y, rng, turn)
    model = aggregate_config["model"]
    tokenizer = aggregate_config["tokenizer"]
    config = aggregate_config["config"]
    max_new_tokens = aggregate_config["max_new_tokens"]
    return generate(model, tokenizer, prompt, config, max_new_tokens)


def build_judge_wo_ref_prompt(hist, y, turn):
    """Return the judge (no ref) prompt string only (for use with generate_batch)."""
    if turn == 1:
        x1, y1 = hist, y
        return f"""{JUDGE_WO_REF_INSTRUCTION}

[Question]
{x1}

[The Start of Assistant's Answer]
{y1}
[The End of Assistant's Answer]"""
    x1, y1S, x2 = hist
    y2 = y
    return f"""{JUDGE_WO_REF_INSTRUCTION}

<|The Start of Assistant A's Conversation with User|>
### User:
{x1}

### Assistant A:
{y1S}

### User:
{x2}

### Assistant A:
{y2}
<|The End of Assistant A's Conversation with User|>"""


def judge_wo_ref(hist, y, turn, judge_config):
    prompt = build_judge_wo_ref_prompt(hist, y, turn)
    model = judge_config["model"]
    tokenizer = judge_config["tokenizer"]
    config = judge_config["config"]
    max_new_tokens = judge_config["max_new_tokens"]
    return generate(model, tokenizer, prompt, config, max_new_tokens)


def build_judge_with_ref_prompt(hist, y, turn):
    """Return the judge (with ref) prompt string only (for use with generate_batch)."""
    if turn == 1:
        x1, r1 = hist
        y1 = y
        return f"""{JUDGE_WITH_REF_INSTRUCTION}

[Question]
{x1}

[The Start of Reference Answer]
{r1}
[The End of Reference Answer]

[The Start of Assistant's Answer]
{y1}
[The End of Assistant's Answer]"""
    x1, r1, y1S, x2, r2 = hist
    y2 = y
    return f"""{JUDGE_WITH_REF_INSTRUCTION}

<|The Start of Reference Answer|>
### User:
{x1}

### Reference answer:
{r1}

### User:
{x2}

### Reference answer:
{r2}
<|The End of Reference Answer|>


<|The Start of Assistant A's Conversation with User|>
### User:
{x1}

### Assistant A:
{y1S}

### User:
{x2}

### Assistant A:
{y2}
<|The End of Assistant A's Conversation with User|>"""


def judge_with_ref(hist, y, turn, judge_config):
    prompt = build_judge_with_ref_prompt(hist, y, turn)
    model = judge_config["model"]
    tokenizer = judge_config["tokenizer"]
    config = judge_config["config"]
    max_new_tokens = judge_config["max_new_tokens"]
    return generate(model, tokenizer, prompt, config, max_new_tokens)


def response_to_rating(response):
    m = re.search(r"\[\[(\d+)\]\]", response)
    return int(m.group(1)) if m else None
