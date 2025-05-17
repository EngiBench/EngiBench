
# 📘 EngiBench Level 1 & Level 2 Dataset Field Descriptions

This document provides detailed descriptions of each field in the Level 1 and Level 2 datasets used in the EngiBench benchmark. These tasks assess foundational knowledge retrieval and contextual reasoning capabilities of large language models (LLMs) on engineering problems.

## 🧾 Field Descriptions

| Field Name                             | Description                                                                                              |
|----------------------------------------|----------------------------------------------------------------------------------------------------------|
| `problem`                              | The original problem statement in natural language.                                                      |
| `answer`                               | The correct numerical or symbolic answer to the original problem.                                        |
| `subfield`                             | Engineering subfield (e.g., Systems & Control, Physical & Structural, Chemical & Biological).            |
| `category`                             | Topical or contextual category of the problem (e.g., Thermodynamics, Circuits, Ocean).                   |
| `difficulty`                           | Difficulty level of the problem: `Level 1` (basic retrieval), `Level 2` (multi-step contextual reasoning). |
| `converted_problem`                    | A mathematically abstracted version of the problem, stripped of domain context.                          |
| `converted_problem_llm_answer`         | The LLM-predicted answer to the `converted_problem`, used to analyze reasoning transfer.                 |
| `knowledge_enhanced_problem`           | A version of the problem augmented with explicit domain knowledge (e.g., constants, formulas).           |
| `rewritten_problem`                    | A semantically or numerically perturbed variant of the original problem for robustness testing.          |
| `rewritten_answer`                     | Correct answer to the `rewritten_problem`.                                                               |
| `rewritten_converted_problem`         | The abstracted mathematical version of the rewritten problem.                                            |
| `rewritten_converted_problem_llm_answer` | LLM-predicted answer to the rewritten converted problem.                                                 |
| `rewritten_knowledge_enhanced_problem` | Rewritten problem version enhanced with domain knowledge, for evaluating reasoning with explicit support. |

## 📌 Notes

- All versions aim to isolate and evaluate specific capabilities such as abstraction, robustness, and knowledge integration.
- Tasks are categorized by domain to support domain-specific analysis.
- Answers and model predictions are used for both automatic and rubric-based evaluation depending on the level.

