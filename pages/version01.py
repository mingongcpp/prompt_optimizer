import streamlit as st
import requests
import os

# ================= CONFIG =================
st.set_page_config(
    page_title="Prompt-Based Classification Optimizer",
    layout="wide"
)

st.title("Prompt-Based Classification Optimizer")
st.write(
    """
    This app converts a **definition-only classification prompt**
    into a **structured, rule-based classification prompt**
    using model disagreement themes.
    """
)

st.write("Version: 2026-02-02")

# ================= API KEY =================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    st.warning("Please set OPENROUTER_API_KEY in Streamlit Secrets.")

# ================= INPUTS =================
st.header("1. Definition-only Prompt")
definition_prompt = st.text_area(
    "Paste the definition-only prompt here:",
    height=150
)

st.header("2. Model Disagreement Themes")
disagreement_themes = st.text_area(
    "Paste the disagreement themes and representative statements here:",
    height=320
)

# ================= SYSTEM PROMPT (STRICT) =================
SYSTEM_PROMPT = """
You are a research assistant helping to optimize a classification prompt
to maximize intercoder reliability.

Your task is to automatically revise a definition-only classification prompt
into a structured, rule-based classification prompt.

CRITICAL OUTPUT CONSTRAINTS:
- You MUST follow the EXACT XML structure provided.
- Do NOT add, remove, rename, or reorder XML tags.
- Use NUMBERED inclusion and exclusion criteria (1, 2, 3, ...).
- Each example MUST be wrapped in its own <Example_n> tag.
- Do NOT repeat example numbers.
- Do NOT introduce any tags that are not specified.
- The <Output_Format> section is REQUIRED and must be included exactly as specified.
- Do NOT include explanations outside XML tags.

DESIGN GOAL:
The goal is NOT completeness, but the most straightforward rules that
disambiguate edge cases and improve agreement among coders.

You MUST output ONLY valid XML that strictly follows the requested structure.
"""

# ================= OPENROUTER CALL =================
def call_openrouter(model_name, system_prompt, user_prompt):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://streamlit.io",
        "X-Title": "Prompt-Based Classification Optimizer"
    }

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0
    }

    response = requests.post(url, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# ================= OPTIMIZATION =================
st.header("3. Generate Structured Classification Prompts")

if st.button("Generate Structured Prompts"):
    if not definition_prompt or not disagreement_themes:
        st.error("Please provide both the definition-only prompt and disagreement themes.")
    elif not OPENROUTER_API_KEY:
        st.error("Missing OpenRouter API key.")
    else:
        with st.spinner("Generating structured prompts with two models..."):

            user_prompt = f"""
DEFINITION-ONLY PROMPT:
{definition_prompt}

MODEL DISAGREEMENT THEMES AND EXAMPLES:
{disagreement_themes}

Please generate a FINAL, ADJUDICATIVE classification prompt using EXACTLY
the following XML structure. This prompt will be used directly by human
research assistants and LLMs for binary classification, so structural
precision is required.

<classification_prompt>
<Role>...</Role>

<Definition>...</Definition>

<Input>...</Input>

<Task>...</Task>

<Classification>
- tactic_prediction = 1 if the tactic is present
- tactic_prediction = 0 if the tactic is not present
</Classification>

<Inclusion_Criteria>
1. ...
2. ...
3. ...
</Inclusion_Criteria>

<Exclusion_Criteria>
1. ...
2. ...
3. ...
</Exclusion_Criteria>

<Examples>
<Example_1>...</Example_1>
<Example_2>...</Example_2>
<Example_3>...</Example_3>
</Examples>

<Output_Format>
Return your answer as JSON only, using exactly this schema:
{{
  "reasoning": "inclusion_criteria [] and exclusion_criteria []; explanation: [step-by-step reasoning with keywords]",
  "tactic_prediction": 0 or 1
}}
</Output_Format>

</classification_prompt>
"""

            col1, col2 = st.columns(2)

            # ===== GPT-5.2 =====
            with col1:
                st.subheader("GPT-5.2 Revised Prompt")
                try:
                    gpt_prompt = call_openrouter(
                        model_name="openai/gpt-5.2",
                        system_prompt=SYSTEM_PROMPT,
                        user_prompt=user_prompt
                    )
                    st.text_area(
                        "Structured Classification Prompt (GPT-5.2):",
                        gpt_prompt,
                        height=560
                    )
                except Exception as e:
                    st.error(f"GPT-5.2 error: {e}")

            # ===== CLAUDE OPUS 4.5 =====
            with col2:
                st.subheader("Claude Opus 4.5 Revised Prompt")
                try:
                    claude_prompt = call_openrouter(
                        model_name="anthropic/claude-opus-4.5",
                        system_prompt=SYSTEM_PROMPT,
                        user_prompt=user_prompt
                    )
                    st.text_area(
                        "Structured Classification Prompt (Claude Opus 4.5):",
                        claude_prompt,
                        height=560
                    )
                except Exception as e:
                    st.error(f"Claude error: {e}")
