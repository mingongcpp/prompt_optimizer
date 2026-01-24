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
    This app automatically converts a **definition-only classification prompt**
    into a **structured and detailed classification prompt** using model
    disagreement themes and example statements.
    """
)

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
    height=300
)

# ================= SYSTEM PROMPT =================
SYSTEM_PROMPT = """
You are a research assistant helping to optimize a classification prompt
to improve intercoder reliability.

Your task is to automatically revise a definition-only classification prompt
into a structured, detailed classification prompt.

Use the disagreement themes and example statements to:
1. Identify key ambiguities in classification.
2. Convert each ambiguity into clear inclusion criteria and exclusion criteria.
3. Select representative positive and negative examples that clarify edge cases.
4. Output a structured classification prompt using XML tags.

The goal is NOT to be comprehensive, but to provide the most straightforward
rules that reduce ambiguity and improve agreement among coders.

You MUST output only the revised classification prompt in valid XML format.
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
        with st.spinner("Generating structured prompts using two models..."):

            user_prompt = f"""
DEFINITION-ONLY PROMPT:
{definition_prompt}

MODEL DISAGREEMENT THEMES AND EXAMPLES:
{disagreement_themes}

Please generate a revised classification prompt using the following XML structure:

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
[List and explain inclusion criteria]
</Inclusion_Criteria>
<Exclusion_Criteria>
[List and explain exclusion criteria]
</Exclusion_Criteria>
<Examples>
<Example_1>...</Example_1>
<Example_2>...</Example_2>
<Example_3>...</Example_3>
</Examples>
</classification_prompt>
"""

            col1, col2 = st.columns(2)

            # ===== GPT =====
            with col1:
                st.subheader("GPT-based Revised Prompt")
                try:
                    gpt_prompt = call_openrouter(
                        model_name="openai/gpt-4.1",
                        system_prompt=SYSTEM_PROMPT,
                        user_prompt=user_prompt
                    )
                    st.text_area(
                        "Structured Classification Prompt (GPT):",
                        gpt_prompt,
                        height=500
                    )
                except Exception as e:
                    st.error(f"GPT error: {e}")

            # ===== CLAUDE =====
            with col2:
                st.subheader("Claude-based Revised Prompt")
                try:
                    claude_prompt = call_openrouter(
                        model_name="anthropic/claude-opus-4.5",
                        system_prompt=SYSTEM_PROMPT,
                        user_prompt=user_prompt
                    )
                    st.text_area(
                        "Structured Classification Prompt (Claude):",
                        claude_prompt,
                        height=500
                    )
                except Exception as e:
                    st.error(f"Claude error: {e}")
