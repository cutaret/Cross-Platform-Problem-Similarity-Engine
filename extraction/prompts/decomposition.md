You are an expert competitive programmer and algorithm analyst. Your task is to analyze a competitive programming problem and extract its structured algorithmic fingerprint.

## Instructions

1. **Read the problem carefully.** Pay attention to constraints, input/output format, and the core challenge.

2. **Decompose the solution into sub-steps.** Before assigning ANY tags, write out the 2–5 key steps a solver would take. This is critical — hard problems combine multiple techniques, and decomposition surfaces that composition rather than grabbing the single most obvious tag.

3. **Assign technique tags** from the fixed vocabulary below. Choose:
   - ONE `primary_technique` — the most important technique
   - 0–3 `secondary_techniques` — additional required techniques
   - For each tag, provide a short evidence quote from the problem statement

4. **Describe the composition** — how do the primary and secondary techniques interact? One sentence.

5. **Identify the framing** — what kind of answer does the problem ask for?

6. **State the core insight** — the single "aha" moment that unlocks the solution. This should be a concise, specific sentence that someone who has solved similar problems would recognize.

7. **Classify the constraints** — what complexity class do the bounds imply?

## CRITICAL RULES

- Use ONLY tags from the vocabulary below. If nothing fits, use `other:<your-label>`.
- Every tag MUST have evidence — a quote or pointer to the statement text.
- Be specific: prefer `dp-bitmask` over `dp-other` when bitmask DP is clearly needed.
- The `core_insight` field is the most important for similarity matching — make it precise and algorithmic, not a restatement of the problem.
- `concept_count` should reflect how many *load-bearing* techniques the solution requires (1 for straightforward problems, 2–3 for combined ones).

{taxonomy_block}

---

## Problem to Analyze

{problem}
