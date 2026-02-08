import streamlit as st
import requests
import os
import pandas as pd

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
    It supports heterogeneous client data formats and coordinates multiple LLMs to explore
    existing marketing theory, map theory to transcripts, and synthesize theory-grounded constructs.
    """
)

# ===============================
# API KEY (OpenRouter)
# ===============================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    st.warning("Please set OPENROUTER_API_KEY in Streamlit Secrets.")

# ===============================
# INPUT: ANY FILE TYPE
# ===============================
st.header("1. Upload Sample Data File")

uploaded_file = st.file_uploader(
    "Upload any file containing conversational text (CSV, TXT, MD, etc.)",
    type=None
)

chat_data = None

if uploaded_file is not None:
    file_name = uploaded_file.name.lower()

    try:
        # -------- CSV HANDLING --------
        if file_name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)

            st.write("Preview of uploaded CSV:")
            st.dataframe(df.head())

            text_column = st.selectbox(
                "Select the column containing conversational text:",
                df.columns
            )

            chat_data = "\n\n".join(
                df[text_column]
                .dropna()
                .astype(str)
                .tolist()
            )

        # -------- OTHER FILE TYPES --------
        else:
            raw_bytes = uploaded_file.read()

            try:
                chat_data = raw_bytes.decode("utf-8")
            except UnicodeDecodeError:
                chat_data = raw_bytes.decode("latin-1", errors="ignore")

        if chat_data:
            st.success("File loaded and converted to text successfully.")
            st.text_area(
                "Preview of extracted text (first 3,000 characters):",
                chat_data[:3000],
                height=200
            )

    except Exception as e:
        st.error(f"Unable to process this file type: {e}")

# ===============================
# PROMPTS
# ===============================
THEORY_EXPLORATION_PROMPT = """
You are a research assistant conducting theory-guided construct exploration
in marketing and sales.

Task:
1. Identify relevant domain-specific marketing and sales theories
   used to explain conversational sales and customer decision-making.
2. Conduct grounded analysis on the provided conversational text.
3. Identify recurring agent behaviors that influence customer commitment.
4. Map these behaviors to theory-grounded constructs.

Requirements:
- Focus on domain-specific theories (e.g., adaptive selling, procedural justice).
- Do NOT treat surface linguistic features as constructs.
- Identify 3–6 theory-grounded constructs.
- Explain how each construct appears in the text.

Output Structure:
1. Relevant Theories
2. Identified Constructs
3. Theory–Data Mapping
"""

JUDGE_PROMPT = """
You are a senior academic reviewer.

Task:
Compare and synthesize two independent theory exploration outputs.

Please:
- Identify overlapping constructs
- Resolve naming differences
- Highlight theoretically robust and empirically observable constructs
- Identify 2–3 constructs most suitable for downstream measurement
- Generate 2–3 testable hypotheses based on these constructs

Output Structure:
1. Overlapping Constructs
2. Final Selected Constructs
3. Hypotheses
"""

# ===============================
# OPENROUTER CALL
# ===============================
def call_openrouter(model_name, prompt, content):
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
            {"role": "system", "content": prompt},
            {"role": "user", "content": content}
        ],
        "temperature": 0
    }

    response = requests.post(url, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# ===============================
# RUN THEORY EXPLORATION
# ===============================
st.header("2. Run Theory Exploration")

col1, col2 = st.columns(2)

with col1:
    st.subheader("LLM 1: Theory Exploration")
    if st.button("Run Theory Exploration (LLM 1)"):
        if chat_data:
            output_1 = call_openrouter(
                model_name="openai/gpt-4.1",
                prompt=THEORY_EXPLORATION_PROMPT,
                content=chat_data
            )
            st.session_state["output_1"] = output_1
            st.text_area("LLM 1 Output", output_1, height=400)
        else:
            st.error("Please upload a file containing conversational text.")

with col2:
    st.subheader("LLM 2: Theory Exploration")
    if st.button("Run Theory Exploration (LLM 2)"):
        if chat_data:
            output_2 = call_openrouter(
                model_name="google/gemini-1.5-pro",
                prompt=THEORY_EXPLORATION_PROMPT,
                content=chat_data
            )
            st.session_state["output_2"] = output_2
            st.text_area("LLM 2 Output", output_2, height=400)
        else:
            st.error("Please upload a file containing conversational text.")

# ===============================
# JUDGE / SYNTHESIS
# ===============================
st.header("3. Compare & Synthesize (Judge Model)")

if st.button("Run Judge Model (Claude)"):
    if "output_1" in st.session_state and "output_2" in st.session_state:
        combined_input = f"""
OUTPUT 1:
{st.session_state["output_1"]}

OUTPUT 2:
{st.session_state["output_2"]}
"""
        judge_output = call_openrouter(
            model_name="anthropic/claude-opus-4.5",
            prompt=JUDGE_PROMPT,
            content=combined_input
        )
        st.text_area(
            "Judge Output (Final Constructs & Hypotheses)",
            judge_output,
            height=500
        )
    else:
        st.error("Please run theory exploration with both LLMs first.")

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.caption(
    "This app supports theory exploration using heterogeneous client data formats "
    "and identifies theory-grounded constructs prior to measurement."
)
