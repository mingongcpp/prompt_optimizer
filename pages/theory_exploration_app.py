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
    theory, map constructs, and generate hypotheses in a reproducible way.
    """
)

# ===============================
# API KEY
# ===============================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    st.warning("Please set OPENROUTER_API_KEY in Streamlit Secrets.")

# ===============================
# INPUT FILE
# ===============================
st.header("1. Upload Sample Data File")

uploaded_file = st.file_uploader(
    "Upload a file containing conversational text (CSV, TXT, MD, etc.)",
    type=None
)

chat_data = None

if uploaded_file is not None:
    filename = uploaded_file.name.lower()

    try:
        # ---- CSV ----
        if filename.endswith(".csv"):
            df = pd.read_csv(uploaded_file)

            st.write("CSV preview:")
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

        # ---- TXT / MD ----
        elif filename.endswith(".txt") or filename.endswith(".md"):
            chat_data = uploaded_file.read().decode("utf-8", errors="ignore")

        # ---- OTHER TYPES ----
        else:
            st.warning(
                "This file type cannot be previewed reliably. "
                "The app will attempt to extract readable text."
            )
            chat_data = uploaded_file.read().decode("utf-8", errors="ignore")

        if chat_data:
            st.success("File processed successfully.")
            st.text_area(
                "Preview of extracted text (first 2000 characters):",
                chat_data[:2000],
                height=200
            )

    except Exception as e:
        st.error(f"File processing failed: {e}")

# ===============================
# PROMPTS
# ===============================
THEORY_EXPLORATION_PROMPT = """
You are a research assistant conducting theory-guided construct exploration
in marketing and sales.

Below are sample chat transcripts from a conversational sales context.

Your tasks:
1. Identify relevant domain-specific marketing and sales theories.
2. Conduct grounded analysis on the transcripts.
3. Identify recurring agent behaviors.
4. Map behaviors to theory-grounded constructs.

Requirements:
- Focus on domain-specific theories.
- Do NOT treat surface linguistic features as constructs.
- Identify 3–6 constructs.
- Explain how each construct appears in the transcripts.

Output Structure:
1. Relevant Theories
2. Identified Constructs
3. Theory–Data Mapping
"""

JUDGE_PROMPT = """
You are a senior academic reviewer.

Compare and synthesize two theory exploration outputs.

Tasks:
- Identify overlapping constructs
- Resolve naming differences
- Select constructs suitable for downstream measurement
- Generate 2–3 testable hypotheses

Output Structure:
1. Overlapping Constructs
2. Final Selected Constructs
3. Hypotheses
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

    # ❗ Do NOT crash the app
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
# RUN THEORY EXPLORATION
# ===============================
st.header("2. Run Theory Exploration")

col1, col2 = st.columns(2)

with col1:
    st.subheader("LLM 1")
    if st.button("Run Theory Exploration (LLM 1)"):
        if chat_data:
            output_1 = call_openrouter(
                model_name="openai/gpt-4.1",
                system_prompt=THEORY_EXPLORATION_PROMPT,
                content=chat_data
            )
            st.session_state["output_1"] = output_1
            st.text_area("LLM 1 Output", output_1, height=400)
        else:
            st.error("Please upload a data file first.")

with col2:
    st.subheader("LLM 2")
    if st.button("Run Theory Exploration (LLM 2)"):
        if chat_data:
            output_2 = call_openrouter(
                model_name="google/gemini-1.5-pro",
                system_prompt=THEORY_EXPLORATION_PROMPT,
                content=chat_data
            )

            # ---- Fallback ----
            if output_2.startswith("[ERROR]"):
                st.warning("LLM 2 failed. Falling back to GPT-4.1.")
                output_2 = call_openrouter(
                    model_name="openai/gpt-4.1",
                    system_prompt=THEORY_EXPLORATION_PROMPT,
                    content=chat_data
                )

            st.session_state["output_2"] = output_2
            st.text_area("LLM 2 Output", output_2, height=400)
        else:
            st.error("Please upload a data file first.")

# ===============================
# JUDGE / SYNTHESIS
# ===============================
st.header("3. Compare & Synthesize (Judge Model)")

if st.button("Run Judge Model"):
    if "output_1" in st.session_state and "output_2" in st.session_state:
        combined_input = f"""
OUTPUT 1:
{st.session_state["output_1"]}

OUTPUT 2:
{st.session_state["output_2"]}
"""
        judge_output = call_openrouter(
            model_name="anthropic/claude-opus-4.5",
            system_prompt=JUDGE_PROMPT,
            content=combined_input
        )
        st.text_area(
            "Final Constructs & Hypotheses",
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
    "This app supports robust, reproducible theory exploration for method-focused analysis."
)
