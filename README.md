# Reddit Music-Advice Classifier

## Overview

This project classifies public comments from **r/WeAreTheMusicMakers** according to the kind of advice they provide. The classifier predicts one of three labels:

- `explained_actionable_advice`: gives a usable recommendation and meaningfully supports it
- `brief_actionable_advice`: gives a usable recommendation with little explanation
- `non_actionable_response`: does not give a realistic action or next step

I fine-tuned `distilbert-base-uncased` on a manually reviewed dataset of 200 Reddit comments and compared it with a zero-shot Groq baseline. Both models were evaluated on the same held-out test set of 30 comments.

The final fine-tuned model achieved **60.0% accuracy**, compared with **46.7% accuracy** for the baseline, an improvement of **13.33 percentage points**.

---

## Research Question

Can a relatively small transformer model, fine-tuned on a small task-specific dataset, distinguish between explained advice, brief advice, and non-actionable responses more accurately than a general-purpose zero-shot language model?

---

## Community and Classification Task

I chose **r/WeAreTheMusicMakers**, a community where musicians, songwriters, producers, performers, and audio engineers discuss songwriting, recording, production, live performance, and creative workflow.

This community is a good fit for classification because responses differ in an observable but nontrivial way. Some comments provide a concrete action and explain why or how to perform it. Others provide a usable action with little support. Still others are reactions, jokes, vague encouragement, personal stories, or statements that do not give the reader a realistic next step.

I used individual public comments as the classification unit. I excluded comments that could not be understood without the surrounding thread because the model receives only the comment text as input.

---

## Label Definitions

### `explained_actionable_advice`

A comment receives this label when it gives the reader a specific action, technique, or next step **and** meaningfully supports it. Support may take the form of reasoning, implementation details, examples, qualifications, a stated purpose, or relevant firsthand experience.

Length alone does not determine the label. A short response can qualify when it contains both a clear recommendation and meaningful support.

### `brief_actionable_advice`

A comment receives this label when it gives the reader a concrete and usable action but provides little or no explanation for why, when, or how the recommendation should be used.

The action must still be specific enough for the reader to attempt. Merely naming an activity or using an imperative verb does not automatically make a comment actionable.

### `non_actionable_response`

A comment receives this label when it does not provide a realistic action or next step. This includes reactions, jokes, vague encouragement, personal anecdotes without a recommendation, dismissive remarks, and commands that are too vague to guide behavior.

---

## Dataset

The dataset contains **200 public Reddit comments** collected from multiple r/WeAreTheMusicMakers threads. Topics include songwriting, recording, mixing and mastering, vocals and instruments, music theory, live performance, creative workflow, and collaboration.

Each row contains:

| Column   | Description                                    |
| -------- | ---------------------------------------------- |
| `text`   | The Reddit comment                             |
| `label`  | The manually assigned classification label     |
| `notes`  | A brief explanation of the annotation decision |
| `source` | The Reddit source when available               |

### Class Distribution

| Label                         |   Count |
| ----------------------------- | ------: |
| `explained_actionable_advice` |      93 |
| `brief_actionable_advice`     |      65 |
| `non_actionable_response`     |      42 |
| **Total**                     | **200** |

The distribution is somewhat imbalanced, especially between `explained_actionable_advice` and `non_actionable_response`. This matters because a model can achieve moderate overall accuracy while still performing poorly on the smaller class.

More detailed annotation rules and edge-case decisions are documented in [`planning.md`](planning.md).

---

## Annotation Process

I first asked whether a comment gave the reader a usable next step. If it did not, I labeled it `non_actionable_response`.

When a usable action was present, I then asked whether the comment meaningfully explained how, why, or when to use the recommendation:

- Action plus meaningful support → `explained_actionable_advice`
- Action with little support → `brief_actionable_advice`
- No realistic action → `non_actionable_response`

I did not use comment length as a direct labeling rule. A short comment could contain a recommendation and a reason, while a long comment could discuss a topic without providing a usable next step.

AI assisted with preliminary labeling, but I manually reviewed and corrected every example before training. The final labels were my decisions rather than unreviewed AI outputs.

### Difficult Annotation Boundaries

The hardest boundary was between explained and brief advice. Explanations were sometimes implicit in a sequence, purpose, or condition rather than introduced by an explicit phrase such as “because.”

Another difficult boundary was between brief and non-actionable responses. Informal comments often used command verbs such as “try,” “sample,” or “experiment,” but some of these comments remained too vague to give the reader a usable procedure.

A final challenge was consistency. A workflow description could appear actionable even when its original annotation marked it as non-actionable. These cases showed that written definitions must be applied consistently across similar examples.

---

## Models

### Fine-Tuned Classifier

The supervised classifier uses:

```text
distilbert-base-uncased
```

Label mapping:

```text
explained_actionable_advice = 0
brief_actionable_advice = 1
non_actionable_response = 2
```

I chose DistilBERT because it is smaller and faster than full BERT while retaining pretrained contextual language representations that are useful for text classification. This made it practical for a small project while still allowing task-specific fine-tuning.

### Zero-Shot Baseline

The baseline uses a Groq-hosted language model prompted with the same three label definitions. It predicts a label without being trained on this project’s annotated examples.

The baseline provides a useful comparison because it tests whether fine-tuning on a small, task-specific dataset improves performance over a general-purpose model following written instructions.

---

## Evaluation Design

I evaluated both models on the same held-out test set of **30 comments**. The support for each class was:

- 14 `explained_actionable_advice`
- 10 `brief_actionable_advice`
- 6 `non_actionable_response`

I report accuracy, precision, recall, F1 score, macro averages, weighted averages, and a confusion matrix.

Accuracy shows the overall percentage of correct predictions, but it is not sufficient because the classes are imbalanced. Macro F1 gives each class equal weight and therefore reveals whether performance is balanced. Per-class precision and recall show which labels the model overpredicts or misses. The confusion matrix shows the exact directions of the errors.

---

## Training Iterations

The initial fine-tuned model achieved **43.3% accuracy**, which was worse than the zero-shot baseline.

| Label                         | Precision |   Recall |       F1 | Support |
| ----------------------------- | --------: | -------: | -------: | ------: |
| `explained_actionable_advice` |      0.48 |     0.86 |     0.62 |      14 |
| `brief_actionable_advice`     |      0.20 |     0.10 |     0.13 |      10 |
| `non_actionable_response`     |      0.00 |     0.00 |     0.00 |       6 |
| **Macro average**             |  **0.23** | **0.32** | **0.25** |  **30** |
| **Weighted average**          |  **0.29** | **0.43** | **0.33** |  **30** |

The initial model heavily favored `explained_actionable_advice`. It correctly identified many explained comments but failed to identify any non-actionable responses.

I made three targeted changes to the existing training configuration:

- Reduced the warmup period because `warmup_steps=50` occupied too much of the training process for a small dataset.
- Reduced the training batch size to 8 so the model received more parameter updates per epoch.
- Increased training to 6 epochs so the model had more opportunities to learn the label boundaries.

After these changes, fine-tuned accuracy increased from **43.3% to 60.0%**.

---

## Final Results

### Overall Accuracy

| Model                   |    Accuracy | Correct Predictions | Test Examples |
| ----------------------- | ----------: | ------------------: | ------------: |
| Zero-shot Groq baseline |      0.4667 |                  14 |            30 |
| Fine-tuned DistilBERT   |      0.6000 |                  18 |            30 |
| **Improvement**         | **+0.1333** |              **+4** |             — |

The fine-tuned model correctly classified four more comments than the baseline.

Because the test set contains only 30 examples, each prediction changes accuracy by approximately **3.33 percentage points**. The improvement is meaningful for this experiment, but it should not be treated as a stable estimate of real-world performance without evaluation on a larger test set.

### Baseline Per-Class Metrics

| Label                         | Precision |   Recall |       F1 | Support |
| ----------------------------- | --------: | -------: | -------: | ------: |
| `explained_actionable_advice` |      0.57 |     0.93 |     0.70 |      14 |
| `brief_actionable_advice`     |      0.33 |     0.10 |     0.15 |      10 |
| `non_actionable_response`     |      0.00 |     0.00 |     0.00 |       6 |
| **Macro average**             |  **0.30** | **0.34** | **0.29** |  **30** |
| **Weighted average**          |  **0.37** | **0.47** | **0.38** |  **30** |

The baseline strongly favored `explained_actionable_advice`. It identified 93% of the explained examples but almost completely failed on the other two labels.

### Final Fine-Tuned Per-Class Metrics

| Label                         | Precision |   Recall |       F1 | Support |
| ----------------------------- | --------: | -------: | -------: | ------: |
| `explained_actionable_advice` |      0.69 |     0.79 |     0.73 |      14 |
| `brief_actionable_advice`     |      0.46 |     0.60 |     0.52 |      10 |
| `non_actionable_response`     |      1.00 |     0.17 |     0.29 |       6 |
| **Macro average**             |  **0.72** | **0.52** | **0.51** |  **30** |
| **Weighted average**          |  **0.67** | **0.60** | **0.57** |  **30** |

The fine-tuned model improved substantially on `brief_actionable_advice`, raising its recall from 0.10 to 0.60. It also made one correct `non_actionable_response` prediction and did not falsely predict that label, producing perfect precision but very low recall. The model remained much better at recognizing advice than recognizing the absence of usable advice.

### Fine-Tuned Confusion Matrix

Rows represent true labels. Columns represent predicted labels.

| True label ↓ / Predicted label → | Explained actionable | Brief actionable | Non-actionable |
| -------------------------------- | -------------------: | ---------------: | -------------: |
| **Explained actionable**         |               **11** |            **3** |          **0** |
| **Brief actionable**             |                **4** |            **6** |          **0** |
| **Non-actionable**               |                **1** |            **4** |          **1** |

The row totals are 14, 10, and 6, for an overall total of 30 examples.

---

## Sample Classifications

The following examples were run through the final fine-tuned model.

| Example text                                                                                                                                                                                                    | Predicted label               | Confidence | Correct?                                                  |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------- | ---------: | --------------------------------------------------------- |
| “Make the vocal sound good without effects first. Add effects such as reverb and delay later. Listen to the vocal within the full mix while adjusting the EQ. A high-pass filter should remove low rumble...”   | `brief_actionable_advice`     |       0.71 | No — dataset label: `explained_actionable_advice`         |
| “throw shit at a wall and see what sticks, sample chop resample etc”                                                                                                                                            | `brief_actionable_advice`     |       0.58 | No — true label: `non_actionable_response`                |
| “Begin with a simple demo containing basic harmony, rhythm, and a scratch vocal. During recording, experiment with sound choices, layering, melodic additions, and how sparse or full each section should...”   | `brief_actionable_advice`     |       0.84 | No — true label: `non_actionable_response` in the dataset |
| “Adding layers is the wrong approach. Figure out what you want to do with the song and what are the minimum amount of elements to do that. For example, a good way to do this is to think of writing the song…” | `explained_actionable_advice` |       0.84 | Yes                                                       |

This prediction is reasonable because the comment gives a specific next step—deciding the song’s intended effect and identifying the minimum elements needed—and supports that recommendation with an example of how to apply it.

---

## Misclassification Analysis

### Explained Advice Predicted as Brief Advice

> Make the vocal sound good without effects first. Add effects such as reverb and delay later. Listen to the vocal within the full mix while adjusting the EQ. A high-pass filter should remove low rumble...

- **True label:** `explained_actionable_advice`
- **Predicted label:** `brief_actionable_advice`
- **Confidence:** 0.71

The model correctly recognized that the response was actionable but underestimated the amount of support it provided. The comment gives an ordered vocal-mixing workflow and states a direct purpose for the high-pass filter: removing low rumble.

The explanation is embedded in the procedure rather than written as an explicit causal sentence. The model may have learned to associate explained advice with longer or more obvious reasoning and missed explanations expressed through purposes, conditions, or ordered steps.

### Vague Response Predicted as Brief Advice

> throw shit at a wall and see what sticks, sample chop resample etc

- **True label:** `non_actionable_response`
- **Predicted label:** `brief_actionable_advice`
- **Confidence:** 0.58

The comment contains action words such as “sample,” “chop,” and “resample,” but it does not specify what to sample, how to change it, what problem the process addresses, or what outcome would count as successful.

The model likely relied on imperative language and music-production vocabulary. It recognized possible activities but did not distinguish between naming an activity and giving a usable recommendation.

### Possible Ground-Truth Inconsistency

> Begin with a simple demo containing basic harmony, rhythm, and a scratch vocal. During recording, experiment with sound choices, layering, melodic additions, and how sparse or full each section should...

- **True label:** `non_actionable_response`
- **Predicted label:** `brief_actionable_advice`
- **Confidence:** 0.84

This example appears to contain several usable actions, including beginning with a simple demo and considering layering and arrangement density. Under the written definitions, it appears more consistent with one of the actionable labels than with `non_actionable_response`.

The model’s confidence does not prove that its prediction was correct, but the disagreement reveals a possible inconsistency between the annotation and the project’s stated rules. More training data would not fix contradictory labels; the annotation should first be reviewed in its full context.

---

## Reflection: What the Model Captured Versus What I Intended

I intended the classifier to make two conceptual decisions. First, it should decide whether a comment gives the reader a concrete and realistic action. Second, when an action is present, it should decide whether the comment meaningfully supports that action.

The final model captured a rougher boundary. It learned that instructional language, imperative verbs, production terminology, and workflow-like phrasing are evidence that a comment contains advice. This helped it distinguish many actionable comments from general discussion and improved its recognition of `brief_actionable_advice` compared with the initial model and the baseline.

However, the model appears to have overfit to the **surface form of advice** rather than the full definition of actionability. Words such as “begin,” “sample,” “chop,” and “experiment” often pushed a response toward an actionable label even when the instruction was vague. In other words, the model learned that a comment sounds instructional more reliably than it learned whether the reader could actually follow the instruction.

The model also missed forms of explanation that were short or implicit. It did not consistently treat an ordered procedure, stated purpose, qualification, or condition as meaningful support unless the reasoning was especially explicit. This blurred the boundary between `explained_actionable_advice` and `brief_actionable_advice`.

The decision boundary therefore does not fully match the intended labels. The intended task depends on semantic judgments about usability and support, while the learned boundary relies more heavily on lexical cues, command structure, and domain vocabulary. The very low recall for `non_actionable_response` is the clearest evidence of this gap.

---

## How I Would Improve the Model

### Audit Annotation Consistency

I would first review all examples near the label boundaries, especially non-actionable comments containing command verbs, explained advice with short reasoning, and workflow descriptions that may have been labeled inconsistently.

### Add Targeted Hard Examples

Rather than collecting random comments, I would add examples designed to separate the difficult boundaries:

- Vague commands that remain non-actionable
- Specific commands with no explanation
- Short actions with clear reasons
- Long comments that never provide a usable next step
- Informal or profane comments in all three classes
- Step-by-step workflows with and without meaningful support

### Increase the Non-Actionable Class

The dataset contains only 42 non-actionable examples compared with 93 explained examples. More diverse negative examples could help the model avoid treating command verbs or music vocabulary as sufficient evidence of actionable advice.

### Use a Larger Evaluation Set

A 30-example test set produces unstable estimates, particularly for a class with only six examples. A larger held-out set or repeated stratified splits would provide a more reliable picture of performance.

### Test Class Weighting and Multiple Splits

Class-weighted loss may place more importance on mistakes involving the smaller classes. Repeated random splits or cross-validation would also show whether the improvement is consistent rather than dependent on one favorable partition.

---

## Success Criteria

The original targets were:

- At least 75% overall accuracy
- Macro F1 of at least 0.70
- F1 of at least 0.65 for every label
- Better accuracy or macro F1 than the zero-shot baseline

The final model achieved:

```text
Fine-tuned accuracy: 60.0%
Baseline accuracy:   46.7%
Fine-tuned macro F1: 0.51
Baseline macro F1:   0.29
```

The model met the goal of outperforming the baseline in both accuracy and macro F1. It did not meet the targets of 75% accuracy, 0.70 macro F1, or at least 0.65 F1 for every label. Only `explained_actionable_advice` exceeded the per-class F1 target.

The result supports the claim that task-specific fine-tuning helped, but the classifier is not accurate or balanced enough for reliable deployment.

---

## Spec Reflection

One way the specification helped was by requiring explicit label definitions and difficult edge cases before training. That forced me to define the task as two decisions—whether a usable action exists and whether it is meaningfully supported—instead of relying on an informal sense of whether a comment was “helpful.” Those definitions also gave me a framework for interpreting the confusion matrix and identifying why the model confused brief, explained, and non-actionable comments.

My implementation diverged from the original plan in its final training configuration. The initial hyperparameters produced only 43.3% accuracy and strongly favored the largest class, so I reduced the warmup period and batch size and increased the number of epochs. I made these changes because the original configuration did not provide enough useful parameter updates for a dataset of only 200 examples. The final configuration remained within the same DistilBERT fine-tuning approach but adapted the implementation to the observed behavior of the model.

---

## AI Usage

I used AI assistance in several specific, reviewable ways.

### Preliminary Annotation Assistance

I directed an AI tool to assign preliminary labels to collected Reddit comments using my three label definitions. It produced suggested labels that accelerated the first pass through the dataset. I did not accept those labels automatically. I manually reviewed every example, compared the suggestion with the written decision rules, and corrected labels when the AI treated vague commands as actionable or failed to recognize meaningful support.

### Error-Pattern Analysis

After the final evaluation, I directed an AI tool to examine the incorrect predictions and look for recurring patterns involving confused labels, comment length, informal language, vague commands, and possible annotation inconsistencies. It suggested that the model relied heavily on action verbs and that some disagreements might reflect inconsistent ground truth. I then reread the examples myself, retained the patterns supported by the actual predictions, and rejected the assumption that every disagreement was necessarily a model failure.

### README and Metric Formatting

I directed an AI tool to help organize the final classification report, confusion matrix, and evaluation discussion into readable Markdown. It produced draft tables and prose. I checked the table entries against the reported support, precision, recall, and F1 values; corrected stale placeholder language; and revised the interpretation so that it accurately stated which success criteria the final model did and did not meet.

AI was not used as an unreviewed source of final labels, metrics, or conclusions. I remained responsible for the annotations, code changes, interpretation, and final written report.

---

## Limitations

The dataset is small and comes from a single Reddit community. The classifier may not generalize to other subreddits, professional audio forums, educational writing, or spoken advice.

The labels also require judgment. The distinction between brief and explained advice depends on what counts as meaningful support, while the distinction between vague and usable advice depends partly on the expected knowledge of the reader.

Reddit comments frequently contain informal grammar, profanity, sarcasm, missing context, domain-specific terminology, and multiple ideas within one response. These characteristics create additional ambiguity.

The model evaluates the structure of advice, not its factual correctness. A technically incorrect recommendation can still be classified as explained actionable advice when it gives a clear action and supports it.

---

## Responsible Use

This classifier could be used for organization, search, or recommendation. It should not be used to automatically punish users or remove comments.

A prediction does not determine whether someone is a good musician, whether the commenter is qualified, or whether the advice is technically correct. It only estimates whether the response appears to contain explained advice, brief advice, or no usable next step.
