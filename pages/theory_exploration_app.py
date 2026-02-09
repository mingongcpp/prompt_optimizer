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
    "Paste sample chat transcripts here (10–15 conversations recommended):",
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

Your task is to compare and synthesize two independent theory exploration outputs.

Please produce a clear, human-readable synthesis using the following format ONLY.

## Final Theory-Grounded Constructs

Present a table with the following columns:
- Construct Name
- Observable Behavior (example or description from transcripts)
- Theory
- Typical Outcome

## Key Hypotheses

List 2–3 concise, testable hypotheses (H1, H2, H3 if applicable) that explain how the constructs influence conversational outcomes.

Do NOT output XML or JSON.
Do NOT include explanations outside the sections above.
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

# -------- LLM 1 --------
with col1:
    st.subheader("LLM 1 (GPT-5.2-chat)")
    if st.button("Run LLM 1"):
        if chat_data:
            st.session_state["output_1"] = call_openrouter(
                "openai/gpt-5.2-chat",
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

# -------- LLM 2 --------
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
                st.warning("LLM 2 failed. Falling back to GPT-5.2-chat.")
                result = call_openrouter(
                    "openai/gpt-5.2-chat",
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
# JUDGE MODEL
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

# -------- Display Judge Output --------
if "judge_output" in st.session_state:
    st.markdown(st.session_state["judge_output"])

# ===============================
# PARSE JUDGE TABLE → CSV
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

        df = pd.DataFrame(rows, columns=headers)

        st.subheader("Parsed Constructs Table")
        st.dataframe(df)

        st.download_button(
            label="Download Constructs as CSV",
            data=df.to_csv(index=False),
            file_name="theory_exploration_constructs.csv",
            mime="text/csv"
        )
    else:
        st.info("No table detected in judge output.")

# ===============================
# EXPORT ALL RESULTS (ARCHIVE)
# ===============================
st.header("5. Download Full Results (Archive)")

export_content = ""

if "output_1" in st.session_state:
    export_content += "\n\n=== LLM 1 OUTPUT ===\n\n" + st.session_state["output_1"]

if "output_2" in st.session_state:
    export_content += "\n\n=== LLM 2 OUTPUT ===\n\n" + st.session_state["output_2"]

if "judge_output" in st.session_state:
    export_content += "\n\n=== JUDGE OUTPUT ===\n\n" + st.session_state["judge_output"]

if export_content:
    st.download_button(
        label="Download Full Results (TXT)",
        data=export_content,
        file_name="theory_exploration_results.txt",
        mime="text/plain"
    )

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.caption(
    "This app supports persistent multi-model theory exploration with CSV-exportable synthesis."
)
