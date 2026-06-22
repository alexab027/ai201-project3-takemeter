# Reddit Music-Advice Classification Project

## Community

I chose **r/WeAreTheMusicMakers**, an online community where musicians, songwriters, producers, performers, and audio engineers discuss songwriting, recording, production, live performance, and the creative process. I chose this community because I have experience with music and recording, which will help me understand the terminology and context of the comments and make informed labeling decisions.

This community is a good fit for classification because responses differ in an observable way: some give a specific action and explain it, some give an action with little explanation, and others do not provide a usable next step. I will use individual public comments as the classification unit and will only include comments that can be understood without the full surrounding thread.

## Labels

### `explained_actionable_advice`

A comment receives this label when it gives the reader a specific action, technique, or next step **and** meaningfully explains or supports it. Support may take the form of reasoning, implementation details, examples, qualifications, or relevant firsthand expertise.

Length, bullet points, and professional credentials do not determine the label by themselves. The response must contain both a usable recommendation and meaningful support.

**Example 1**

you can't remove sibilance with basic EQ. You need dynamic EQ or de-esser for that and even then it's a band-aid. You want to ensure there's no sibilance to begin with. Change the mic position, change the mice type, change how you speak. Pop filters do not protect against sibilance, they protect against plosives. Tons of sibilance also is cause by ALIASING which means bad audio chain and bad decisions even before the file came to your desk.

**Example 2**

I mute all the tracks, then bring things back in one at a time starting with the "most important". For me, I usually bring drums and vocals back in first. Make sure I'm happy with the balance between just those and it helps me catch anything wrong with either of those two. Then I'll add bass back in and try and get that balanced with the drums and vocals. Then any other stuff in, one at a time, checking after each to see when I goes wrong/ when it gets muddy. Sometimes there's too much stuff in a mix and this approach helps me cut some tracks which are actually making the full mix worse.

### `brief_actionable_advice`

A comment receives this label when it gives the reader a relevant and usable action, technique, or next step but provides little explanation, implementation detail, or supporting evidence.

The recommendation must be specific enough to point the reader in a productive direction. A response is not placed here merely because it is positive or related to the topic.

**Example 1**

Learning about note intervals helped me write more interesting melodies. Scales are like the foundation, but intervals is how you get something unique sounding.

**Example 2**

Networking. Organic social and in person. Your nearest city has multiple networks. Find your people. Go to shows. Build relationships with the local studio engineers and producers.

### `non_actionable_response`

A comment receives this label when it does not give the reader a realistic action or next step. It may be a joke, reaction, unsupported hot take, personal preference, dismissal, insult, irrelevant response, or broad observation without practical direction.

A strongly worded opinion can still be actionable if it tells the reader what to do. Likewise, a personal story can be actionable if it produces a clear transferable recommendation.

**Example 1**

play better

**Example 2**

Sorry, but saying you made a song "completely from scratch" by choosing samples in a DAW is like saying you baked by heating up muffins in the microwave.

## Hard Edge Cases

The main boundary is between `explained_actionable_advice` and `brief_actionable_advice`. I will first identify the action the commenter recommends. I will then ask whether the response explains how, why, or when to use that recommendation.

- If the reader receives both a usable action and meaningful support, the label is `explained_actionable_advice`.
- If the reader receives a usable action but little support, the label is `brief_actionable_advice`.
- If there is no clear action, the label is `non_actionable_response`.

Relevant experience counts as support only when it is connected to the recommendation. Saying “I have been a producer for twenty years” is not actionable by itself. Saying “I have mixed live sound for twenty years, and I first lower the instruments before raising the vocal monitor because this reduces feedback” is explained actionable advice.

A short answer can still be explained actionable advice if it clearly states both the action and the reason. A long answer can still be non-actionable if it never gives the reader a usable next step.

Ambiguous cases will be documented in the CSV's `notes` column and reviewed for consistency.

## Data Collection Plan

I collected **200 public comments** from multiple r/WeAreTheMusicMakers threads covering songwriting, recording, mixing, mastering, vocals, instruments, live performance, music theory, workflow, and collaboration.

The dataset is stored in one CSV with these columns:

- `text`: the Reddit comment
- `label`: one of the three classification labels
- `notes`: a brief explanation of the annotation
- `source`: the Reddit source when available

Current distribution:

- `explained_actionable_advice`: 93
- `brief_actionable_advice`: 65
- `non_actionable_response`: 42

I used AI to assist with preliminary labeling, but I will personally review and correct every example before training. If a class remains underrepresented after review, I will collect additional comments targeted toward that category rather than changing labels merely to balance the dataset.

## Evaluation Metrics

I will report overall accuracy, precision, recall, and F1 score for each class. I will also report macro-averaged F1 so that each label contributes equally even if the classes are not perfectly balanced.

A confusion matrix will show which categories the model most often confuses. I expect the most difficult boundary to be between explained and brief actionable advice, since both contain recommendations and differ mainly in the amount of meaningful support.

I will compare the fine-tuned classifier with a zero-shot Groq baseline on the same held-out test set.

## Definition of Success

I will consider the project successful if the fine-tuned model achieves:

- At least **75% overall accuracy**
- A **macro F1 score of at least 0.70**
- An **F1 score of at least 0.65 for every label**
- Better overall accuracy or macro F1 than the zero-shot baseline

For a real deployment, I would require at least 80% accuracy, a macro F1 of at least 0.75, and additional testing on newer threads not represented in training.

The classifier should be used for organization or recommendation rather than automatic punishment or removal. Its prediction describes the structure of a response—whether it contains explained advice, brief advice, or no actionable advice—not whether the commenter is a good musician or whether every technical claim is objectively correct.
