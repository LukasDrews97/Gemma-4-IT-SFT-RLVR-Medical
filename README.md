# Gemma-4-IT-SFT-RLVR-Medical

Gemma-4-it finetuned on [PubMedQA](https://pubmedqa.github.io/), using SFT and RLVR.

### Sample Output with Reasoning (SFT + RLVR): <br>

<b>Question:</b> `Do GEC produce and bear factor H under complement attack?` <br>
<b>Answer: </b>
```
This study was the first to demonstrate that factor H was expressed as a protein on the membrane of GEC. Since GEC are an important target in the development of complement-mediated kidney injury, and factor H is involved in the ability of the kidney to produce factor H, the study thus provides a new perspective on the interactions between factor H and GEC.
Final Decision: yes
```

<details>
<summary><b>View Internal Thinking Process and Detailed Reasoning</b></summary>
1.  **Analyze the Request:** The user has provided a scientific context (a research abstract) and implicitly expects an answer (yes/no/maybe) based on the provided text.

2.  **Analyze the Context (Abstract):**
    *   **Topic:** Factor H in rat glomerular epithelial cells (GEC).
    *   **Methodology:** Used polyclonal antibody, PCR-based cloning, RT-PCR, immunofluorescence microscopy, Western blot, and Heymann nephritis model.
    *   **Key Findings:**
        *   Factor H was identified on cultured GEC membranes.
        *   Cloning GEC factor H cDNA showed it was full-length.
        *   Factor H mRNA increased under complement activation in cultured GEC and in the PHN model.

3.  **Determine the Core Claim/Conclusion:** The GEC produce and express factor H.

4.  **Relate Findings to the Question:** The question asks if GEC produce and bear factor H under complement attack. The findings directly support this conclusion.

5.  **Formulate the Reasoning:** Summarize the key experimental evidence that led to the conclusion. (GEC are a source of factor H.)

6.  **Final Check:** Does the answer address the context accurately and directly? Yes. (The study confirms a new role for GEC in factor H production).Despite its importance in the regulation of complement activation, the main source of factor H in the kidney is still unclear. Our study demonstrates that GEC produce factor H and the synthesized factor H is presented on the surface of these cells. We therefore conclude that GEC may be a source of factor H that contributes to the kidney's ability to resist complement attack. <br>
Final Decision: yes
</details>


## One-command setup with uv

```bash
uv sync --no-dev &&
uv run QLORA_SFT.py && 
uv run GRPO_RLVR.py && 
uv run evaluate.py
```


## Results

All models are evaluated on a holdout evaluation dataset of 1000 samples. Experiments ran on a RTX 5070 Ti.

| **Model**                   | **# SFT samples** | **# RLVR samples** |  **# Evaluation samples** | **PubMedQA Accuracy** |
|-----------------------------|------------------:|-------------------:|----------:|----------------------:|
| Gemma-4-E2B-it (base model) | -                 | -                  | 1000      | 58.10 %               |
| Gemma-4-E2B-it + SFT + RLVR | 1024 (3 epochs)   | 256                | 1000      | 73.10%                |
| Gemma-4-E4B-it (base model) | -                 | -                  | 1000      | Work in Progress      |
| Gemma-4-E4B-it + SFT + RLVR | 1024 (3 epochs)   | 256                | 1000      | Work in Progress      |
