import streamlit as st
from dataclasses import dataclass
from typing import List, Dict, Any
import uuid
import json
import io

# =========================
# 数据结构
# =========================

@dataclass
class TextUnit:
    id: str
    text: str


@dataclass
class TheoryExplorationResult:
    model_name: str
    identified_constructs: List[Dict[str, Any]]
    notes: str


@dataclass
class SynthesisResult:
    synthesized_constructs: List[Dict[str, Any]]
    hypotheses: List[str]


# =========================
# Step 1: 文件读取 & 文本抽取
# =========================

def extract_text_from_file(uploaded_file) -> str:
    """
    从任意上传文件中尽最大努力抽取文本
    （当前为通用 fallback，后续可按格式扩展）
    """

    try:
        # 尝试当作 UTF-8 文本直接读
        bytes_data = uploaded_file.read()
        text = bytes_data.decode("utf-8", errors="ignore")
        return text

    except Exception as e:
        return ""


def load_text_units_from_text(raw_text: str) -> List[TextUnit]:
    """
    将原始文本拆分为最小分析单元（按行）
    """
    lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
    return [
        TextUnit(id=str(uuid.uuid4()), text=line)
        for line in lines
    ]


# =========================
# Step 2: 独立理论探索（Mock）
# =========================

def explore_theory_with_model(
    model_name: str,
    text_units: List[TextUnit]
) -> TheoryExplorationResult:

    constructs = [
        {
            "construct_name": "Perceived Helpfulness",
            "theoretical_origin": "Service-Dominant Logic",
            "behavioral_indicators": [
                "proactive clarification",
                "anticipation of user needs"
            ],
            "example_text_unit_ids": [tu.id for tu in text_units[:2]]
        }
    ]

    notes = (
        f"{model_name} independently explored theory-grounded sales constructs "
        f"based on observed conversational behaviors."
    )

    return TheoryExplorationResult(
        model_name=model_name,
        identified_constructs=constructs,
        notes=notes
    )


# =========================
# Step 3: Judge Model 综合（Mock）
# =========================

def synthesize_with_judge_model(
    results: List[TheoryExplorationResult]
) -> SynthesisResult:

    synthesized_constructs = [
        {
            "construct_name": "Perceived Helpfulness",
            "merged_from_models": [r.model_name for r in results],
            "definition": (
                "The extent to which the agent’s responses reduce user effort "
                "and increase decision clarity."
            ),
            "empirical_observability": "High"
        }
    ]

    hypotheses = [
        "H1: Early demonstrations of perceived helpfulness increase later conversational engagement.",
        "H2: Proactive explanations before persuasive attempts increase user trust signals."
    ]

    return SynthesisResult(
        synthesized_constructs=synthesized_constructs,
        hypotheses=hypotheses
    )


# =========================
# Streamlit UI
# =========================

st.set_page_config(
    page_title="Theory-Guided Construct Exploration",
    layout="wide"
)

st.title("🧠 Theory-Guided Construct Exploration App")

st.markdown(
    """
This app supports **theory-guided construct exploration** for conversational sales data.

**Workflow**
1. Upload heterogeneous client data  
2. Independent theory exploration by multiple models  
3. Judge model synthesis  
4. Generation of testable hypotheses  
"""
)

# -------- Step 1: 上传文件 --------
st.subheader("1️⃣ Upload Conversational Data")

uploaded_file = st.file_uploader(
    "Upload a file (any format: txt, csv, pdf, json, email logs, etc.)",
    type=None
)

run_button = st.button("Run Theory Exploration")

# -------- 主流程 --------
if run_button and uploaded_file is not None:

    with st.spinner("Extracting text from uploaded file..."):
        raw_text = extract_text_from_file(uploaded_file)

    if not raw_text.strip():
        st.error(
            "No readable text could be extracted from this file. "
            "You may need a format-specific parser (e.g., PDF, DOCX)."
        )
        st.stop()

    text_units = load_text_units_from_text(raw_text)

    st.success(
        f"Extracted {len(text_units)} text units from `{uploaded_file.name}`"
    )

    # -------- 独立探索 --------
    with st.spinner("Running independent theory exploration..."):
        result_a = explore_theory_with_model("LLM_A", text_units)
        result_b = explore_theory_with_model("LLM_B", text_units)

    # -------- Judge 综合 --------
    with st.spinner("Synthesizing constructs and hypotheses..."):
        synthesis = synthesize_with_judge_model([result_a, result_b])

    # -------- 输出 --------
    st.subheader("2️⃣ Independent Model Explorations")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Model A Output**")
        st.json(result_a.__dict__)

    with col2:
        st.markdown("**Model B Output**")
        st.json(result_b.__dict__)

    st.subheader("3️⃣ Judge Model Synthesis")

    st.markdown("**Synthesized Constructs**")
    st.json(synthesis.synthesized_constructs)

    st.markdown("**Generated Hypotheses**")
    for h in synthesis.hypotheses:
        st.write("-", h)

    # -------- 导出 --------
    st.subheader("4️⃣ Export Results")

    export_data = {
        "source_file": uploaded_file.name,
        "text_unit_count": len(text_units),
        "independent_explorations": [
            result_a.__dict__,
            result_b.__dict__
        ],
        "synthesis": synthesis.__dict__
    }

    st.download_button(
        label="Download Results as JSON",
        data=json.dumps(export_data, indent=2),
        file_name="theory_exploration_results.json",
        mime="application/json"
    )

elif run_button:
    st.warning("Please upload a file before running the analysis.")
