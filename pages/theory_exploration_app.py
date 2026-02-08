import streamlit as st
from dataclasses import dataclass
from typing import List, Dict, Any
import uuid
import json

# =========================================================
# 数据结构
# =========================================================

@dataclass
class TextUnit:
    id: str
    text: str


@dataclass
class TheoryExplorationResult:
    model_name: str
    raw_model_output: str


@dataclass
class SynthesisResult:
    synthesized_constructs: str
    hypotheses: str


# =========================================================
# Step 1: 文件读取 & 文本抽取
# =========================================================

def extract_text_from_file(uploaded_file) -> str:
    """
    Generic text extraction fallback.
    Assumes heterogeneous client data.
    """
    bytes_data = uploaded_file.read()
    return bytes_data.decode("utf-8", errors="ignore")


def split_into_text_units(raw_text: str) -> List[TextUnit]:
    """
    Split raw text into analyzable conversational units.
    """
    lines = [l.strip() for l in raw_text.split("\n") if l.strip()]
    return [
        TextUnit(id=str(uuid.uuid4()), text=line)
        for line in lines
    ]


# =========================================================
# Step 2: 独立理论探索（ChatGPT 5.2 / Gemini 3.0）
# =========================================================

def run_theory_exploration_llm(
    model_label: str,
    text_units: List[TextUnit]
) -> TheoryExplorationResult:
    """
    Calls an LLM as an independent theory explorer.
    The prompt is written for human inspection and academic transparency.
    """

    conversation_excerpt = "\n".join(
        f"- {tu.text}" for tu in text_units[:50]
    )

    prompt = f"""
You are acting as an independent theory-exploration agent.

Your task is to analyze conversational sales or service text
through the lens of established marketing and sales theories.

You should NOT assume predefined variables.

Instead:

1. Draw on relevant theories (e.g., persuasion, trust, relationship marketing,
   service-dominant logic, signaling, uncertainty reduction).
2. Identify recurring *agent behaviors* observable in the text.
3. Propose theory-grounded constructs that could plausibly influence
   customer decisions.
4. Ground each construct explicitly in patterns from the text.

Do NOT coordinate with other models.
Do NOT attempt synthesis.

Conversational data:
{conversation_excerpt}
"""

    # -----------------------------------------------------
    # IMPORTANT:
    # This is where the real API call would go.
    # The model_label is intentionally explicit for transparency.
    # -----------------------------------------------------

    simulated_output = f"""
[Model: {model_label}]

Identified Construct: Perceived Helpfulness
Theoretical Basis: Service-Dominant Logic; Cognitive Load Reduction

Observed Behavioral Patterns:
- Agent proactively explains options without being prompted.
- Agent anticipates potential confusion points for the user.

Rationale:
These behaviors reduce user effort and increase perceived competence
of the agent, which theory suggests should influence trust and engagement.
"""

    return TheoryExplorationResult(
        model_name=model_label,
        raw_model_output=simulated_output
    )


# =========================================================
# Step 3: Judge Model 综合（Claude 4.5）
# =========================================================

def run_judge_synthesis(
    exploration_results: List[TheoryExplorationResult]
) -> SynthesisResult:
    """
    Claude 4.5 acts as a judge model.
    It does NOT re-analyze raw text.
    It only synthesizes model outputs.
    """

    model_outputs = "\n\n".join(
        f"--- Output from {r.model_name} ---\n{r.raw_model_output}"
        for r in exploration_results
    )

    judge_prompt = f"""
You are acting as a judge model responsible for theory synthesis.

Your task:

1. Compare independent theory exploration outputs from multiple models.
2. Identify overlapping or conceptually equivalent constructs,
   even if naming differs.
3. Retain only constructs that are:
   - Theory-grounded
   - Empirically observable in conversational text
4. Produce a clean, unified construct definition.
5. Generate testable hypotheses, with attention to
   timing and sequencing in conversation.

You MUST NOT introduce new constructs
that were not supported by at least one model.

Independent model outputs:
{model_outputs}
"""

    simulated_synthesis = """
Synthesized Construct:
Perceived Helpfulness

Definition:
The extent to which an agent’s conversational behavior reduces user effort,
anticipates informational needs, and increases decision clarity.

Testable Hypotheses:
H1: Demonstrations of perceived helpfulness early in the conversation
    increase user engagement in subsequent turns.

H2: Proactive explanatory behaviors preceding persuasive attempts
    increase user trust signals.
"""

    return SynthesisResult(
        synthesized_constructs=simulated_synthesis,
        hypotheses=simulated_synthesis
    )


# =========================================================
# Streamlit UI
# =========================================================

st.set_page_config(
    page_title="Theory-Guided Construct Exploration",
    layout="wide"
)

st.title("🧠 Theory-Guided Construct Exploration App")

st.markdown("""
This app formalizes a **theory-first exploration workflow** for conversational data.

**Models**
- Independent Explorers: ChatGPT 5.2, Gemini 3.0  
- Judge Model: Claude 4.5
""")

uploaded_file = st.file_uploader(
    "Upload conversational data (any format)",
    type=None
)

run_button = st.button("Run Theory Exploration")

if run_button and uploaded_file is not None:

    raw_text = extract_text_from_file(uploaded_file)
    text_units = split_into_text_units(raw_text)

    st.success(f"Extracted {len(text_units)} text units.")

    with st.spinner("Running independent theory exploration..."):
        gpt_result = run_theory_exploration_llm(
            "ChatGPT 5.2", text_units
        )
        gemini_result = run_theory_exploration_llm(
            "Gemini 3.0", text_units
        )

    with st.spinner("Running judge model synthesis..."):
        synthesis = run_judge_synthesis(
            [gpt_result, gemini_result]
        )

    st.subheader("Independent Model Outputs")
    st.text(gpt_result.raw_model_output)
    st.text(gemini_result.raw_model_output)

    st.subheader("Judge Model Synthesis (Claude 4.5)")
    st.text(synthesis.synthesized_constructs)

    st.download_button(
        "Download Full Results",
        data=json.dumps({
            "explorers": [
                gpt_result.__dict__,
                gemini_result.__dict__
            ],
            "judge": synthesis.__dict__
        }, indent=2),
        file_name="theory_exploration_results.json"
    )

elif run_button:
    st.warning("Please upload a file first.")
