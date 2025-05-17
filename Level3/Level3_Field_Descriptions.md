
# 📙 EngiBench Level 3 Dataset Field Descriptions

This document provides detailed descriptions of each field in the Level 3 dataset used in the EngiBench benchmark. Level 3 focuses on open-ended, real-world engineering tasks, evaluated using a rubric-based system across four core capabilities.

## 🧾 Field Descriptions

| Field Name                           | Description                                                                                               |
|--------------------------------------|-----------------------------------------------------------------------------------------------------------|
| `question_original_language`         | The original open-ended modeling question in its native language (typically Chinese).                    |
| `question`                           | The translated version of the original question in English.                                               |
| `question_modified`                  | Semantically perturbed version of the original question.                                                  |
| `subquestion_original_language`      | The sub-question(s) used in rubric evaluation, in the original language.                                  |
| `subquestion`                        | English translation of the sub-question(s).                                                               |
| `subquestion_modified`               | Semantically perturbed version of the sub-question.                                                       |
| `source_detail`                      | The original source of the question (e.g., MCM competition, course assignments, national exam).           |
| `official_scoring_standard_original_language` | Scoring rubric in the original language provided by the source institution.                              |
| `official_scoring_standard`          | Translated version of the official rubric into English.                                                   |
| `subfield`                           | Engineering subfield (e.g., Systems & Control, Structural, Chemical & Biological).                        |
| `category`                           | Topical domain or context.                                                               |
| `information_extraction_score`       | Score for the model's ability to identify and filter relevant information from complex inputs.            |
| `multi_objective_decision_score`     | Score for the model's ability to balance competing objectives and make trade-offs.                        |
| `uncertainty_handling_score`         | Score for the model's ability to reason under uncertainty or incomplete data.                             |
| `domain_specific_reasoning_score`    | Score for the model's ability to apply engineering knowledge and perform domain-specific reasoning.       |
