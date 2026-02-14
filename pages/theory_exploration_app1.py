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
    This app operationalizes a **theory-guided construct exploration workflow**
    for textual data in marketing, persuasion, and strategic communication contexts.

    Upload a CSV file containing textual data (must include columns: `id`, `caption`).
    """
)

# ===============================
# API KEY
# ===============================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    st.warning("Please set OPENROUTER_API_KEY in environment variables.")

# ===============================
# STEP 1: UPLOAD CSV
# ===============================
st.header("1. Upload Text Dataset (CSV)")

uploaded_file = st.file_uploader(
    "Upload CSV file (must contain columns: id, caption)",
    type=["csv"]
)

text_data = None
df = None

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        if "id" not in df.columns or "caption" not in df.columns:
            st.error("CSV must contain 'id' and 'caption' columns.")
        else:
            st.success("File uploaded successfully.")
            st.subheader("Preview Data")
            st.dataframe(df.head())

            # Combine captions into one text block for exploration
            text_data = "\n\n".join(
                [f"ID {row['id']}: {row['caption']}" for _, row in df.iterrows()]
            )

    except Exception as e:
        st.error(f"Error reading file: {e}")

# ===============================
# PROMPTS
# ===============================
THEORY_EXPLORATION_PROMPT = """
You are a research assistant conducting theory-guided construct exploration
in marketing, persuasion, and strategic communication research.

Below is a sample of textual data.

Tasks:
1. Identify relevant domain-specific theories (e.g., persuasion theory, influence principles, branding theory, consumer psychology, communication theory).
2. Conduct grounded analysis of the text.
3. Identify recurring communicative or strategic behaviors.
4. Map these behaviors to theory-grounded constructs.

Requirements:
- Focus on theoretically established constructs.
- Do NOT invent new theories.
- Do NOT treat surface linguistic features as constructs.
- Identify 3–6 constructs that are both theory-grounded and observable in the data.
- Clearly explain how each construct manifests in the text.

Output clearly with section headers.
"""

JUDGE_PROMPT = """
You are a senior academic reviewer.

Your task is to compare and synthesize two independent theory exploration outputs.

Please produce a clear, human-readable synthesis using the following format ONLY.

## Final Theory-Grounded Constructs

Present a table with the following columns:
- Construct Name
- Observable Behavior (example or description from text)
- Theory
- Typical Outcome

## Key Hypotheses

List 2–3 concise, testable hypotheses (H1, H2, H3 if applicable)
that explain how the constructs influence communication or persuasion outcomes.

Do NOT output XML or JSON.
Do NOT include explanations outside the sections above.
"""

# ===============================
# OPENROUTER CALL
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
            {"role": "user", "content": f"Here is the textual dataset:\n\n{content}"}
        ],
        "temperature": 0
    }

    response = requests.post(url, headers=headers, json=payload, timeout=120)

    if response.status_code != 200:
        return f"[ERROR] {response.text}"

    try:
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[ERROR] Failed to parse response: {e}"

# ===============================
# STEP 2: LLM EXPLORATION
# ===============================
st.header("2. Run Theory Exploration")

col1, col2 = st.columns(2)

# -------- LLM 1 --------
with col1:
    st.subheader("LLM 1 (GPT-5.2-chat)")
    if st.button("Run LLM 1"):
        if text_data:
            st.session_state["output_1"] = call_openrouter(
                "openai/gpt-5.2-chat",
                THEORY_EXPLORATION_PROMPT,
                text_data
            )
        else:
            st.error("Please upload a valid CSV file first.")

    if "output_1" in st.session_state:
        st.text_area("LLM 1 Output", st.session_state["output_1"], height=350)

# -------- LLM 2 --------
with col2:
    st.subheader("LLM 2 (Gemini 3 Flash)")
    if st.button("Run LLM 2"):
        if text_data:
            result = call_openrouter(
                "google/gemini-3-flash-preview",
                THEORY_EXPLORATION_PROMPT,
                text_data
            )

            if result.startswith("[ERROR]"):
                st.warning("LLM 2 failed. Falling back to GPT-5.2-chat.")
                result = call_openrouter(
                    "openai/gpt-5.2-chat",
                    THEORY_EXPLORATION_PROMPT,
                    text_data
                )

            st.session_state["output_2"] = result
        else:
            st.error("Please upload a valid CSV file first.")

    if "output_2" in st.session_state:
        st.text_area("LLM 2 Output", st.session_state["output_2"], height=350)

# ===============================
# STEP 3: JUDGE MODEL
# ===============================
st.header("3. Judge Model Synthesis")

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
    st.markdown(st.session_state["judge_output"])

# ===============================
# STEP 4: EXPORT TABLE
# ===============================
st.header("4. Export Judge Results as CSV")

if "judge_output" in st.session_state:
    lines = st.session_state["judge_output"].splitlines()

    table_lines = [
        line for line in lines
        if "|" in line and not line.strip().startswith("|---")
    ]

    if len(table_lines) >= 2:
        headers = [h.strip() for h in table_lines[0].split("|")[1:-1]]
        rows = [
            [cell.strip() for cell in row.split("|")[1:-1]]
            for row in table_lines[1:]
        ]

        df_constructs = pd.DataFrame(rows, columns=headers)

        st.subheader("Parsed Constructs Table")
        st.dataframe(df_constructs)

        st.download_button(
            label="Download Constructs as CSV",
            data=df_constructs.to_csv(index=False),
            file_name="theory_exploration_constructs.csv",
            mime="text/csv"
        )
    else:
        st.info("No table detected in judge output.")

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.caption("This app supports multi-model theory-guided construct exploration using uploaded CSV datasets.")
