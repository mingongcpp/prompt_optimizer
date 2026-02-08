import streamlit as st
from dataclasses import dataclass
from typing import List, Dict, Any
import uuid
import json

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
# Step 1: 输入处理
# =========================

def load_text_units(raw_text: str) -> List[TextUnit]:
    """
    将输入文本拆分为最小分析单元（按行）
    """
    lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
    return [
        TextUnit(id=str(uuid.uuid4()), text=line)
        for line in lines
    ]


# =========================
# Step 2: 独立理论探索（占位）
# =========================

def explore_theory_with_model(
    model_name: str,
    text_units: List[TextUnit]
) -> TheoryExplorationResult:
    """
    单模型 theory-guided construct exploration
    （这里是 mock，后续可接 LLM API）
    """

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
        f"{model_name} independently explored marketing and sales theories "
        f"and grounded constructs in conversational behaviors."
    )

    return TheoryExplorationResult(
        model_name=model_name,
        identified_constructs=constructs,
        notes=notes
    )


# =========================
# Step 3: Judge Model 综合（占位）
# =========================

def synthesize_with_judge_model(
    results: List[TheoryExplorationResult]
) -> SynthesisResult:
    """
    Judge model：对齐构念、消解命名差异、生成假设
    """

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
This app operationalizes **theory-guided construct exploration** for conversational sales data.

**Workflow**
1. Upload or paste conversational text  
2. Independent theory exploration by multiple models  
3. Judge model synthesis  
4. Generation of testable hypotheses  
"""
)

# -------- 输入区域 --------
st.subheader("1️⃣ Input Conversational Text")

raw_text = st.text_area(
    "Paste conversational text (one utterance per line):",
    height=200
)

# -------- 运行按钮 --------
run_button = st.button("Run Theory Exploration")

# -------- 主流程 --------
if run_button and raw_text.strip():

    # Step 1
    text_units = load_text_units(raw_text)

    st.success(f"Loaded {len(text_units)} text units.")

    # Step 2
    with st.spinner("Running independent theory exploration..."):
        result_a = explore_theory_with_model("LLM_A", text_units)
        result_b = explore_theory_with_model("LLM_B", text_units)

    # Step 3
    with st.spinner("Synthesizing constructs with judge model..."):
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

    # -------- 可复现导出 --------
    st.subheader("4️⃣ Export Results")

    export_data = {
        "text_units": [tu.__dict__ for tu in text_units],
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
    st.warning("Please paste some conversational text before running.")
