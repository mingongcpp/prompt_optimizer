from typing import List, Dict, Any
from dataclasses import dataclass
import json
import uuid


# ========== 数据结构定义 ==========

@dataclass
class TextUnit:
    """最小分析单元（一句、一轮对话等）"""
    id: str
    text: str


@dataclass
class TheoryExplorationResult:
    """单个模型的理论探索结果"""
    model_name: str
    identified_constructs: List[Dict[str, Any]]
    notes: str


@dataclass
class SynthesisResult:
    """Judge model 的综合输出"""
    synthesized_constructs: List[Dict[str, Any]]
    hypotheses: List[str]


# ========== Step 1: 输入数据处理 ==========

def load_text_units(raw_texts: List[str]) -> List[TextUnit]:
    """
    将异构输入转为统一的可分析文本单元
    """
    return [
        TextUnit(id=str(uuid.uuid4()), text=text)
        for text in raw_texts
    ]


# ========== Step 2: 独立理论探索 ==========

def explore_theory_with_model(
    model_name: str,
    text_units: List[TextUnit]
) -> TheoryExplorationResult:
    """
    使用单个 LLM 进行理论驱动的构念探索
    （此处为占位逻辑，实际可接 OpenAI / Claude / 本地模型）
    """

    # --- 伪代码：实际应为 prompt + LLM 调用 ---
    identified_constructs = [
        {
            "construct_name": "Perceived Helpfulness",
            "theoretical_origin": "Service-Dominant Logic",
            "behavioral_indicators": [
                "agent proactively explains options",
                "agent anticipates user concerns"
            ],
            "example_text_unit_ids": [tu.id for tu in text_units[:2]]
        }
    ]

    notes = (
        f"{model_name} focused on marketing and sales theories "
        f"related to persuasion, trust, and conversational guidance."
    )

    return TheoryExplorationResult(
        model_name=model_name,
        identified_constructs=identified_constructs,
        notes=notes
    )


# ========== Step 3: Judge Model 综合 ==========

def synthesize_with_judge_model(
    exploration_results: List[TheoryExplorationResult]
) -> SynthesisResult:
    """
    Judge model 对多个模型输出进行对齐、去重和理论筛选
    """

    # --- 伪代码：实际应为 judge prompt + LLM 调用 ---
    synthesized_constructs = [
        {
            "construct_name": "Perceived Helpfulness",
            "merged_from_models": [
                r.model_name for r in exploration_results
            ],
            "definition": (
                "The extent to which the agent’s responses reduce "
                "user effort and increase decision clarity."
            ),
            "empirical_observability": "High"
        }
    ]

    hypotheses = [
        (
            "H1: When perceived helpfulness is demonstrated early "
            "in the conversation, user engagement increases in later turns."
        ),
        (
            "H2: Sequencing proactive explanations before pricing information "
            "leads to higher trust signals from users."
        )
    ]

    return SynthesisResult(
        synthesized_constructs=synthesized_constructs,
        hypotheses=hypotheses
    )


# ========== Step 4: 全流程 Orchestration ==========

def run_theory_exploration_pipeline(raw_texts: List[str]) -> Dict[str, Any]:
    """
    将整个 theory → synthesis → hypothesis 的流程串起来
    """

    # Step 1: 处理输入
    text_units = load_text_units(raw_texts)

    # Step 2: 多模型独立探索
    model_a_result = explore_theory_with_model("LLM_A", text_units)
    model_b_result = explore_theory_with_model("LLM_B", text_units)

    # Step 3: Judge 综合
    synthesis = synthesize_with_judge_model(
        [model_a_result, model_b_result]
    )

    # Step 4: 输出结构化结果（便于存档 & 复现）
    output = {
        "text_unit_count": len(text_units),
        "independent_explorations": [
            model_a_result.__dict__,
            model_b_result.__dict__
        ],
        "synthesis": synthesis.__dict__
    }

    return output


# ========== 示例运行 ==========

if __name__ == "__main__":
    sample_texts = [
        "I can help you compare different plans based on your needs.",
        "Most customers in your situation prefer this option."
    ]

    results = run_theory_exploration_pipeline(sample_texts)

    print(json.dumps(results, indent=2, ensure_ascii=False))
