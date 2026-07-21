# 🧬 Comprehensive Fine-Tuned Model Evaluation Framework
**Author:** Lead AI Architect & Prompt Engineer (5x PhD: ML, NLP, Cognitive Science, Statistical Inference, HCI)

Evaluating a fine-tuned LLM requires a multi-dimensional, statistically rigorous approach. Fine-tuning often introduces issues like **semantic drift**, **catastrophic forgetting**, **over-fitting**, or **format degradation**. 

To verify if your fine-tuned model is working "perfectly," we utilize an **LLM-as-a-Judge** meta-evaluation pattern. This system compares the fine-tuned model's outputs against target references and system constraints using strict rubrics.

---

## 1. The Meta-Evaluation Prompt (LLM-as-a-Judge)

Use the prompt below with a highly capable evaluator model (e.g., Gemini 1.5 Pro or Gemini 3.5 Flash) to score individual test inputs and responses.

```markdown
You are an elite AI Evaluation Auditor with PhDs in Computational Linguistics, Statistical Learning, and Cognitive Systems. Your task is to perform a rigorous forensic audit on the output of a recently fine-tuned language model (the "Candidate Model") compared to the desired "Ground Truth" (or Target Reference) and the original "System Prompt Constraints."

### Evaluation Paradigm
You must evaluate the Candidate Model's output across 5 distinct dimensions:
1. **Instruction & Constraint Adherence** (Syntax, structure, negative constraints)
2. **Task Completeness & Accuracy** (Information correctness, completeness)
3. **Style & Tone Alignment** (Tone, vocabulary, pacing, formatting consistency)
4. **Formatting & Structural Integrity** (JSON/XML syntax, schema validation)
5. **Hallucination & Semantic Drift** (Introduction of ungrounded or contradictory info)

---

### INPUT DATA FOR EVALUATION

[SYSTEM INSTRUCTIONS]
{{SYSTEM_INSTRUCTIONS}}
[/SYSTEM INSTRUCTIONS]

[TEST INPUT / PROMPT]
{{TEST_INPUT}}
[/TEST INPUT / PROMPT]

[GROUND TRUTH / TARGET REFERENCE] (Note: Use for relative assessment. If blank, evaluate absolutely against System Instructions)
{{GROUND_TRUTH}}
[/GROUND TRUTH / TARGET REFERENCE]

[CANDIDATE MODEL OUTPUT]
{{CANDIDATE_OUTPUT}}
[/CANDIDATE MODEL OUTPUT]

---

### EVALUATION RUBRIC & SCORING SYSTEM

Evaluate each dimension on a scale of 1 to 5. 

#### Dimension 1: Instruction & Constraint Adherence
*   **5 (Excellent):** Followed all system rules, positive instructions, and negative constraints (e.g., "do not do X").
*   **4 (Good):** Minor deviation, but fully obeyed all negative constraints and core logic.
*   **3 (Fair):** Obeyed core prompt but violated minor constraints or failed on complex logical boundaries.
*   **2 (Poor):** Violated major negative constraints or completely missed key instructions.
*   **1 (Critical Failure):** Totally ignored the system instructions.

#### Dimension 2: Task Completeness & Accuracy
*   **5 (Excellent):** The response is 100% correct, matches the core details of the ground truth, and completes all sub-tasks.
*   **4 (Good):** Correct information, but missing minor details or context from the ground truth.
*   **3 (Fair):** Partially complete; contains some inaccuracies or omissions.
*   **2 (Poor):** Major gaps in completeness or contains significant factual errors.
*   **1 (Critical Failure):** The task was not completed, or the output is completely wrong/hallucinated.

#### Dimension 3: Style & Tone Alignment
*   **5 (Excellent):** The voice matches the target tone (e.g. professional, concise, medical, empathetic) perfectly. The lexicon, length, and flow are indistinguishable from the reference style.
*   **4 (Good):** Tone is generally correct, but minor phrasing is slightly off-style.
*   **3 (Fair):** Functional, but lacks the specific stylistic characteristics required by the fine-tuning target.
*   **2 (Poor):** Inconsistent tone; sounds robotic, overly wordy, or uses inappropriate language style.
*   **1 (Critical Failure):** Completely fails the target tone; sounds like a stock pre-trained base model.

#### Dimension 4: Formatting & Structural Adherence
*   **5 (Excellent):** Strict adherence to format (e.g., pristine JSON, correct XML tags, proper Markdown) without escaping errors or syntax issues. Parseable by code.
*   **4 (Good):** Valid format structure, but contains trivial issues like trailing whitespace or minor formatting quirks.
*   **3 (Fair):** Syntactically correct but did not structure fields exactly as specified (e.g. used camelCase instead of snake_case).
*   **2 (Poor):** Invalid format structure (e.g., unparseable JSON, unclosed brackets, missing required root keys).
*   **1 (Critical Failure):** Returned raw free-form text instead of the requested structured output.

#### Dimension 5: Hallucination & Semantic Drift
*   **5 (Excellent):** Zero hallucination. Every assertion is strictly grounded in the test input or reference documentation. No "knowledge leakage" of unrelated facts.
*   **4 (Good):** Fully grounded, but introduced generic, harmless filler words.
*   **3 (Fair):** Introduced assumptions that are not directly supported by the context, though not explicitly contradicted.
*   **2 (Poor):** Minor hallucination of facts or details not present in the input.
*   **1 (Critical Failure):** Fabricated completely false facts, numbers, or details that conflict with the ground truth.

---

### INSTRUCTIONS FOR YOUR ANALYSIS (THINKING PROCESS)
1. **Forensic Analysis:** Extract all positive/negative constraints from the system instructions. Map them one-by-one to the Candidate Output.
2. **Diff Comparison:** Compare the Candidate Output to the Ground Truth/Target Reference. Note semantic, structural, and stylistic differences.
3. **Chain-of-Thought Justification:** For *each* dimension, write a detailed justification of the score you are giving. State specific parts of the output that led to that score.
4. **Final Decision:** Formulate an overall verification decision: `PASS`, `PASS_WITH_WARNINGS` (if average score is between 3.5 and 4.5, or a minor constraint failed), or `FAIL` (if any score is <= 2, or overall average is < 3.5).

### OUTPUT SCHEMA
You MUST return your output in JSON format only. Do not include markdown code block markers or any leading/trailing text outside the JSON. Use the following schema:

```json
{
  "chain_of_thought": {
    "constraints_checked": [
      {
        "constraint": "string description",
        "adhered": true/false,
        "evidence": "string explanation"
      }
    ],
    "ground_truth_diff": "detailed comparative notes",
    "hallucination_audit": "notes on grounding"
  },
  "scores": {
    "instruction_adherence": { "score": 1-5, "rationale": "string" },
    "task_completeness": { "score": 1-5, "rationale": "string" },
    "style_tone": { "score": 1-5, "rationale": "string" },
    "formatting": { "score": 1-5, "rationale": "string" },
    "hallucination_drift": { "score": 1-5, "rationale": "string" }
  },
  "summary": {
    "average_score": 0.0,
    "status": "PASS | PASS_WITH_WARNINGS | FAIL",
    "failure_reasons": ["list of reasons why it failed or has warnings, empty if PASS"],
    "actionable_feedback": "specific instructions for model refinement"
  }
}
```
```

---

## 2. Automating the Evaluation Pipeline (Python)

To run this at scale, we use a Python script to iterate over an evaluation dataset (Golden Dataset), call the candidate model, send inputs to the Judge LLM, and calculate aggregate metrics.

Here is a ready-to-run automation script using the `google-genai` / `google-generativeai` SDK:

```python
import json
import logging
from typing import Any, Dict, List
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ModelEvaluation")

# Configure GenAI Client (Make sure GEMINI_API_KEY is set in your env)
genai.configure()

JUDGE_PROMPT_TEMPLATE = """
[SYSTEM INSTRUCTIONS]
{system_instructions}
[/SYSTEM INSTRUCTIONS]

[TEST INPUT / PROMPT]
{test_input}
[/TEST INPUT / PROMPT]

[GROUND TRUTH / TARGET REFERENCE]
{ground_truth}
[/GROUND TRUTH / TARGET REFERENCE]

[CANDIDATE MODEL OUTPUT]
{candidate_output}
[/CANDIDATE MODEL OUTPUT]
"""

class ModelEvaluator:
    def __init__(self, judge_model_name: str = "gemini-1.5-pro"):
        """
        Args:
            judge_model_name: The highly capable model acting as the judge.
        """
        # Load the base system instruction for the judge
        # (This is the Meta-Evaluation Prompt from Section 1)
        self.judge_system_instruction = (
            "You are an elite AI Evaluation Auditor with PhDs in Computational Linguistics, "
            "Statistical Learning, and Cognitive Systems. Your task is to perform a rigorous "
            "forensic audit on the output of a recently fine-tuned language model compared to the "
            "desired Ground Truth and original System Prompt Constraints. Follow the provided rubrics "
            "and output strictly valid JSON."
        )
        self.model = genai.GenerativeModel(
            model_name=judge_model_name,
            system_instruction=self.judge_system_instruction,
            generation_config={"response_mime_type": "application/json"}
        )

    def evaluate_case(
        self,
        system_instructions: str,
        test_input: str,
        ground_truth: str,
        candidate_output: str
    ) -> Dict[str, Any]:
        """Evaluates a single test case using the Judge LLM."""
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            system_instructions=system_instructions,
            test_input=test_input,
            ground_truth=ground_truth,
            candidate_output=candidate_output
        )
        
        try:
            response = self.model.generate_content(prompt)
            result = json.loads(response.text)
            return result
        except Exception as e:
            logger.error(f"Error evaluating test case: {e}")
            return {
                "scores": {},
                "summary": {
                    "status": "ERROR",
                    "failure_reasons": [str(e)]
                }
            }

    def run_suite(self, test_cases: List[Dict[str, str]]) -> Dict[str, Any]:
        """Runs evaluation over a list of test cases and generates aggregated metrics."""
        results = []
        passed_count = 0
        total_scores = {
            "instruction_adherence": 0.0,
            "task_completeness": 0.0,
            "style_tone": 0.0,
            "formatting": 0.0,
            "hallucination_drift": 0.0
        }
        
        for idx, case in enumerate(test_cases):
            logger.info(f"Evaluating case {idx + 1}/{len(test_cases)}...")
            eval_res = self.evaluate_case(
                system_instructions=case["system_instructions"],
                test_input=case["test_input"],
                ground_truth=case["ground_truth"],
                candidate_output=case["candidate_output"]
            )
            eval_res["case_metadata"] = case.get("metadata", {})
            results.append(eval_res)
            
            if eval_res.get("summary", {}).get("status") in ["PASS", "PASS_WITH_WARNINGS"]:
                passed_count += 1
                
            scores = eval_res.get("scores", {})
            for key in total_scores:
                total_scores[key] += scores.get(key, {}).get("score", 0)
                
        num_cases = len(test_cases)
        avg_scores = {k: v / num_cases for k, v in total_scores.items()} if num_cases > 0 else {}
        
        return {
            "dataset_summary": {
                "total_cases": num_cases,
                "passed_cases": passed_count,
                "pass_rate": (passed_count / num_cases) * 100 if num_cases > 0 else 0,
                "average_scores": avg_scores
            },
            "detailed_results": results
        }

# Example Usage
if __name__ == "__main__":
    # Define a sample test case representing a CRM brief generator model
    sample_dataset = [
        {
            "metadata": {"case_id": "TC-001"},
            "system_instructions": (
                "Write a concise consultant-facing brief for a clinic CRM. "
                "Do not diagnose. Mention that human review is required."
            ),
            "test_input": "Image type: skin. Flags: ['acne']. Treatments: [{'name': 'Salicylic Acid Peel', 'price': 120}].",
            "ground_truth": (
                "Review suggested for skin consultation showing signs of acne. "
                "Recommend discussing Salicylic Acid Peel ($120). Human review is required."
            ),
            # This represents output from your fine-tuned model under evaluation
            "candidate_output": (
                "We detected severe Acne and recommend a Salicylic Acid Peel. "
                "This will cure the patient's condition. Cost: $120. Needs clinical confirmation."
            )
        }
    ]
    
    evaluator = ModelEvaluator(judge_model_name="gemini-1.5-pro")
    report = evaluator.run_suite(sample_dataset)
    print(json.dumps(report, indent=2))
```

---

## 3. Best Practices for Professional Verification

When evaluating fine-tuned models:
1. **Split Dataset Representation:** Make sure your evaluation dataset contains **In-Distribution (ID)** cases, **Out-of-Distribution (OOD)** cases, and **Negation/Safety** edge cases.
2. **Establish a Baseline:** Run the evaluator prompt on your base model (pre-fine-tuned) and the fine-tuned model to quantify the exact net improvements and regressions.
3. **Audit the Judge:** Run a sanity check on the Judge LLM by feeding it intentionally flawed candidate outputs. Ensure it triggers `FAIL` and lists the correct negative reasons.
4. **Use Structured Verification (CI/CD):** Integrate the automated evaluator into your build or deployment scripts (e.g., using `pytest`) to ensure no model is pushed to production without passing a minimum threshold of score metrics (e.g., `average_score >= 4.2`).
