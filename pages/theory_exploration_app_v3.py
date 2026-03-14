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

st.title("Theory-Guided Sales Tactic Exploration")
st.write(
    """
    This app operationalizes a **theory-guided construct exploration workflow**
    for identifying **theory-grounded sales and persuasive tactics** in textual data.

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
# PROMPTS (UPDATED FOR THEORY-GROUNDED SALES TACTICS)
# ===============================
THEORY_EXPLORATION_PROMPT = """
You are a research assistant conducting THEORY-FIRST construct exploration
for theory-grounded sales and persuasive tactics in brand caption text.

Below is a dataset of Instagram captions.

Goal:
Start from well-established theories/models in marketing, persuasion, and sales,
then derive a small set of theory-grounded sales or persuasive tactics that are
observable in caption text and can be measured with binary coding
(present = 1, absent = 0).

Important conceptual boundary:
- Focus on tactics that reflect persuasive or selling moves enacted in language.
- Do NOT focus on broad content topics, aesthetic style, general tone, or overall caption quality.
- Do NOT output themes such as "product launch," "community post," or "lifestyle content" unless they clearly function as an observable persuasive tactic.
- The construct should capture HOW persuasion/selling is enacted in the caption, not just WHAT the caption is about.

Tasks:
1) Identify 3–6 relevant, established theories/models that plausibly explain
   how caption language uses sales or persuasive tactics to influence audience
   response, action, trust, or engagement.
2) Using ONLY the caption text (ignore visuals), identify recurring messaging
   behaviors that map to these theories.
3) Propose 6 FINAL constructs that are:
   - clearly tied to a specific theory/model,
   - explicitly observable in caption text,
   - suitable for binary coding,
   - likely to have enough positive cases for analysis (avoid extremely rare constructs),
   - actionable for brand communication.

Naming rule (IMPORTANT):
- Each construct must be ONE atomic concept only.
- Do NOT use combined names like "X & Y", "X and Y", "X/Y", "X + Y".
- If two ideas appear together, split them into separate constructs.

Output format:

## Theories/Models Used
List 3–6 theories/models with 1–2 lines on why they fit this dataset.

## Final Theory-Grounded Constructs (Ranked by Expected Prevalence)
Provide a table ranked from most frequent to least frequent with columns:
- Rank
- Construct Name (single concept only)
- Theory Anchor (name a specific theory/model)
- Expected Prevalence (High / Medium / Low)
- Textual Cues for Coding (3–6 short cues)
- Caption Examples (2–3 short excerpts from the dataset)

## Key Hypotheses
List 2–3 testable hypotheses linking these constructs to post outcomes
(e.g., engagement-related metrics).
"""

JUDGE_PROMPT = """
You are a senior academic reviewer.

Your task is to compare and synthesize two independent theory exploration outputs.
Prioritize constructs that are:
(1) observable in caption text,
(2) suitable for binary coding,
(3) likely to be frequent enough to measure,
(4) clearly framed as theory-grounded sales or persuasive tactics,
not broad topics or overall stylistic qualities.

Conceptual boundary:
- Focus on HOW persuasion or selling is enacted in language.
- Exclude broad content topics, general brand tone, aesthetic style, and holistic quality judgments.

SINGLE-CONSTRUCT RULE (VERY IMPORTANT):
- Each construct must be ONE atomic concept only.
- Do NOT output combined constructs such as "X & Y", "X and Y", "X/Y", or "X + Y".
- If necessary, split combined constructs into separate constructs.

Please produce a clear synthesis using the following format ONLY.

## Final Theory-Grounded Constructs (Ranked by Expected Prevalence)

Present a table with the following columns:
- Rank
- Construct Name (single concept only)
- Observable Behavior (from caption text)
- Theory
- Expected Prevalence (High/Medium/Low)
- Textual Cues for Coding (3–6 cues)
- Typical Outcome

## Key Hypotheses
List 2–3 concise, testable hypotheses linking constructs to measurable outcomes (e.g., engagement-related metrics).

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
# STEP 5: DOWNLOAD FULL RESULTS
# ===============================
st.header("5. Download Full Results (Archive)")

export_content = ""

if "output_1" in st.session_state:
    export_content += "\n\n==============================\n"
    export_content += "LLM 1 OUTPUT (GPT-5.2-chat)\n"
    export_content += "==============================\n\n"
    export_content += st.session_state["output_1"]

if "output_2" in st.session_state:
    export_content += "\n\n==============================\n"
    export_content += "LLM 2 OUTPUT (Gemini 3 Flash)\n"
    export_content += "==============================\n\n"
    export_content += st.session_state["output_2"]

if "judge_output" in st.session_state:
    export_content += "\n\n==============================\n"
    export_content += "JUDGE MODEL OUTPUT (Claude Opus)\n"
    export_content += "==============================\n\n"
    export_content += st.session_state["judge_output"]

if export_content:
    st.download_button(
        label="Download Full Results (TXT)",
        data=export_content,
        file_name="theory_exploration_full_results.txt",
        mime="text/plain"
    )
else:
    st.info("Run the models first to generate downloadable results.")


# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.caption("This app supports multi-model theory-guided construct exploration for theory-grounded sales tactics using uploaded CSV datasets.")
