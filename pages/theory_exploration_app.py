import streamlit as st
import requests
import os
from xml.dom import minidom

# ===============================
# CONFIG
# ===============================
st.set_page_config(
    page_title="Theory Exploration App",
    layout="wide"
)

st.title("Theory-Guided Construct Exploration")
st.write(
    """
    This app operationalizes a **theory exploration workflow** for conversational sales data.
    It uses multiple LLMs to independently explore theory and a judge model to synthesize
    theory-grounded constructs and hypotheses in a reproducible pipeline.
    """
)

# ===============================
# API KEY
# ===============================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    st.warning("Please set OPENROUTER_API_KEY in Streamlit Secrets.")

# ===============================
# INPUT TRANSCRIPTS
# ===============================
st.header("1. Enter Sample Chat Transcripts")

chat_data = st.text_area(
    "Paste sample chat transcripts here:",
    height=300,
    placeholder="Customer: I'm not sure about the price.\nAgent: Let me check with my supervisor..."
)

# ===============================
# PROMPTS
# ===============================
THEORY_EXPLORATION_PROMPT = """
You are a research assistant conducting theory-guided construct exploration
in marketing and sales.

Below are sample chat transcripts from a conversational sales context.

Tasks:
1. Identify relevant domain-specific marketing and sales theories.
2. Conduct grounded analysis on the transcripts.
3. Identify recurring agent behaviors.
4. Map behaviors to theory-grounded constructs.

Requirements:
- Focus on domain-specific theories.
- Do NOT treat surface linguistic features as constructs.
- Identify 3–6 constructs.
- Explain how each construct appears in the transcripts.

Output clearly with section headers.
"""

JUDGE_PROMPT = """
You are a senior academic reviewer.

Compare and synthesize two theory exploration outputs.

Output STRICT XML ONLY using the following structure:

<theory_synthesis>
  <final_constructs>
    <construct>
      <name></name>
      <behavior></behavior>
      <theory></theory>
      <outcome></outcome>
    </construct>
  </final_constructs>
  <hypotheses>
    <hypothesis></hypothesis>
  </hypotheses>
</theory_synthesis>
"""

# ===============================
# OPENROUTER CALL (SAFE)
# ===============================
def call_openrouter(model_name, system_prompt, content):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://streamlit.io",
        "X-Title": "Theory Exploration App"
    }

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Here are the chat transcripts:\n\n{content}"
            }
        ],
        "temperature": 0
    }

    response = requests.post(url, headers=headers, json=payload, timeout=120)

    if response.status_code != 200:
        return (
            f"[ERROR]\n"
            f"Model: {model_name}\n"
            f"Status code: {response.status_code}\n"
            f"Response: {response.text}"
        )

    try:
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[ERROR] Failed to parse response: {e}"

# ===============================
# LLM EXPLORATION
# ===============================
st.header("2. Run Theory Exploration")

col1, col2 = st.columns(2)

with col1:
    st.subheader("LLM 1 (GPT-4.1)")
    if st.button("Run LLM 1"):
        if chat_data:
            st.session_state["output_1"] = call_openrouter(
                "openai/gpt-4.1",
                THEORY_EXPLORATION_PROMPT,
                chat_data
            )
        else:
            st.error("Please paste chat transcripts first.")

    if "output_1" in st.session_state:
        st.text_area(
            "LLM 1 Output",
            st.session_state["output_1"],
            height=350
        )

with col2:
    st.subheader("LLM 2 (Gemini 3 Flash)")
    if st.button("Run LLM 2"):
        if chat_data:
            result = call_openrouter(
                "google/gemini-3-flash-preview",
                THEORY_EXPLORATION_PROMPT,
                chat_data
            )

            if result.startswith("[ERROR]"):
                st.warning("LLM 2 failed. Falling back to GPT-4.1.")
                result = call_openrouter(
                    "openai/gpt-4.1",
                    THEORY_EXPLORATION_PROMPT,
                    chat_data
                )

            st.session_state["output_2"] = result
        else:
            st.error("Please paste chat transcripts first.")

    if "output_2" in st.session_state:
        st.text_area(
            "LLM 2 Output",
            st.session_state["output_2"],
            height=350
        )

# ===============================
# JUDGE MODEL (XML)
# ===============================
st.header("3. Judge Model Synthesis (XML Output)")

if st.button("Run Judge Model"):
    if "output_1" in st.session_state and "output_2" in st.session_state:
        combined_input = f"""
OUTPUT 1:
{st.session_state["output_1"]}

OUTPUT 2:
{st.session_state["output_2"]}
"""
        st.session_state["judge_output"] = call_openrouter(
            "anthropic/claude-opus-4.5",
            JUDGE_PROMPT,
            combined_input
        )
    else:
        st.error("Please run both LLM explorations first.")

if "judge_output" in st.session_state:
    st.text_area(
        "Judge Output (XML)",
        st.session_state["judge_output"],
        height=400
    )

# ===============================
# EXPORT RESULTS
# ===============================
st.header("4. Export Results")

def pretty_xml(xml_str):
    try:
        reparsed = minidom.parseString(xml_str)
        return reparsed.toprettyxml(indent="  ")
    except Exception:
        return xml_str

export_content = ""

if "output_1" in st.session_state:
    export_content += "\n\n=== LLM 1 OUTPUT ===\n\n" + st.session_state["output_1"]

if "output_2" in st.session_state:
    export_content += "\n\n=== LLM 2 OUTPUT ===\n\n" + st.session_state["output_2"]

if "judge_output" in st.session_state:
    export_content += "\n\n=== JUDGE OUTPUT (XML) ===\n\n" + pretty_xml(
        st.session_state["judge_output"]
    )

if export_content:
    st.download_button(
        label="Download All Results",
        data=export_content,
        file_name="theory_exploration_results.txt",
        mime="text/plain"
    )

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.caption(
    "This app supports persistent multi-model theory exploration with structured XML synthesis."
)
