# Proposal changes: bringing the Team 27 proposal up to date

**Compared against:** `DAP Project Proposal Team 27.pdf` (19 pages) and the repository as of 10 October 2026.

**Why it needs changes:** the simulator changed after the proposal's numbers and charts were made. Every EDA number in the PDF comes from the old simulator, and several of them are now wrong. This file lists what to change, section by section, with suggested wording you can paste and edit.

Numbers marked **[from notebook]** come from `notebooks/DAP_IEDA_Jenga.ipynb`. Fill them in after generating the new dataset and running the notebook.

---

## 1. What changed since the PDF was written

| What | In the PDF | Now |
|---|---|---|
| Physics time step | 2 ms | **1 ms**. At 2 ms an idle tower slowly slid apart (the top of a gapped tower drifted about 1 mm per second). The drift ate into the tilt test, so many "medium" labels were partly caused by drift. A new automated test fails if this ever comes back. |
| Tilt test resolution | 0.15° | **0.0375°** (4× finer). With the drift gone, most removals change the tilt margin by less than 0.15°, so the coarse test could not see them. |
| Medium threshold | 0.3°, justified as "2× the test's resolution" | Still **0.3°**, now justified by measurement: we re-measured 20 blocks 5 times each with microscopic nudges, and the drop varied by **0.10°** on average. 0.3° is 3× that noise. |
| Labelling bug | (not known) | **Fixed.** Blocks whose drop was exactly 0.3° were labelled low, because in computer arithmetic 5.1 − 4.8 = 0.29999… In the old dataset this affected **1,026 blocks (7.6%)**, all of which should have been medium. A consistency check in the EDA notebook found it. |
| Imperfect towers | not mentioned | **Built.** Half of new towers lean slightly (up to 0.4°), have shifted and twisted layers, and have a few blocks sticking out 2–10 mm, like a real tower a few rounds in. `towers.csv` records `imperfect`, `lean_deg` and `protruding`. |
| Generation speed | "about one tower per minute" | **About 2 minutes per tower per worker**, roughly 0.5 towers a minute with 5 workers (the smaller time step doubled it). |
| Dataset | 300 towers | **Planned: 500 towers** = 400 for training and validation, plus a **separate, frozen 100-tower test set** made with a different seed. Not generated yet. |
| Splitting work across machines | not possible | `--shard` lets the M4 and the Ryzen each build half, then merge. |
| Tests | 62 | **65** |
| EDA notebook | "to write" (Milestone 1) | **Built:** `notebooks/DAP_IEDA_Jenga.ipynb` follows the DAP template's 8 questions and saves every chart to `docs/eda_v2/`. |

**What this means for the numbers:** the old 55 / 19 / 26% split, the 0.46 medium F1, the 81.9% baseline accuracy and every chart will change after regeneration. A preliminary 10-tower check under the new physics, at the 0.3° threshold, gave roughly **59% low / 14% medium / 27% high**. That check was made before the rounding fix, so the final medium share will be somewhat higher. Use the notebook's numbers, not these.

---

## 2. Must fix (wrong or stale facts)

| Section (page) | Current text | Change to |
|---|---|---|
| 7, row 1 (p. 6) | "More towers on demand, about one per minute." | "More towers on demand, about one every 2 minutes per CPU core; one overnight run on two laptops makes 500." |
| 7, row 1 (p. 6) | "300 towers, 1,800 photos (6 views each), 13,532 labelled blocks." | "500 towers (400 training and validation + 100 frozen test), 3,000 photos, about **[from notebook]** labelled blocks. Half the towers are deliberately imperfect." |
| 7, row 1 (p. 6) | "Training and validation for both stages and the end-to-end CNN." | Keep, but add: "The 100-tower test set is generated separately and never looked at until final evaluation." |
| 8, left column (p. 7) | "Simulator, dataset generator, label viewer and 62 tests." | "Simulator, dataset generator, label viewer, EDA notebook and 65 tests." |
| 8, key numbers (p. 7–8) | 300 / 1,800 / 13,532 / 98.5% / 0.46 | Replace all with **[from notebook]** values (sections 2, 5 and 8 of the notebook). |
| 8, findings (p. 15) | "about one tower a minute on a laptop, 300 towers in 4.8 hours" | "about 2 minutes per tower per CPU core; 500 towers in one overnight run across two laptops" |
| 8, findings (p. 15) | "Collapse is predictable. We repeated 480 removals…" | Keep (still true), and add the noise finding: "Medium labels have measurable noise: re-measuring the same removal varies by about 0.1°, so we set the medium threshold at 0.3°, three times that." |
| 8, chart captions (p. 9–14) | All numbers in captions | Replace with the regenerated charts (section 6 below). |
| 10, compute row (p. 17) | "Making data takes about a minute per tower" / "We already have 300 towers" | "About 2 minutes per tower per core. We split each run across two laptops with `--shard`, overnight." |

---

## 3. Template compliance

| Section | Template rule | Problem | Fix |
|---|---|---|---|
| 1. Team | "Include your current commitments" | Commitments column is empty for Lakshan and Gabriel, and "-" for Lee Fan | Fill in each person's winter and term commitments (CCAs, jobs, internships, exams). Mentors use this to judge whether the plan is realistic. |
| 3. Mentors | "Reason / relevant expertise" | June's and Sumin's reasons are only their project titles | Say *why* each fits. Suggested: **June:** "Built a 3D instance-segmentation pipeline, the same technique as our stage 1 block finder." **Sumin:** "Built a 3D simulator, which matches our physics simulator and our sim-to-real question." |
| 4.3 | "Write 1-3 questions" | There are **4** | Merge or drop one. Suggested: drop the look-ahead question (make look-ahead a stretch goal), or merge it into question 2. |
| 9 | "Aim to have Milestone 2 completed by around Week 5 of Term 2" | Met (Sem 2 Week 3) | No change. |
| 11 | "Team Notes (availability, exams, etc.)" | Column is empty | At minimum: "Exams from Sem 2 Week 9" on Weeks 9–11, and anyone's busy weeks. |

---

## 4. Section-by-section changes

### 1. Team
- Fill in **Commitments** for all three members (see section 3 above).
- Gabriel's objective: "add to portfolio ig lol" reads as too casual for a document mentors score. Suggested: "Gain project experience: learn a real project workflow and build my portfolio."

### 2. Group learning objectives
They're good. Two small changes:
- Objective 3 says "beats a **physics-formula** baseline". That baseline (centre of mass vs support) isn't built yet. Either keep it and add it to Milestone 1, or change to "beats our rule-of-thumb and lookup-table baselines, especially on medium risk".
- Make objective 4 measurable: "held-out test towers never seen during development, per-class F1, and results reported separately for tidy and imperfect towers."

### 4.1 Problem statement
Good as is. Add one sentence after the third paragraph:
> "Real towers are also messy: they lean, and blocks get left sticking out. Our simulator now builds half its towers that way, so the model learns from towers that look like a game in progress, not just perfect ones."

### 4.2 Proposed approach
- **Item 1 (simulated data):** add:
  > "Half the towers are built slightly imperfect (a small lean, shifted layers, a few blocks sticking out). The tilt test is precise to 0.04°, and medium starts at a 0.3° drop, three times the measured noise of the test."
- **Item 4 (look-ahead):** "using reinforcement learning as an extension to determine this" mixes the method and the extension. Suggested:
  > "Look-ahead (stretch goal). Using the classifier, look a few moves ahead to estimate the chance of a collapse soon, catching blocks that are safe now but dangerous later. Reinforcement learning is a further extension if time allows."
- **Item 5 (end-to-end CNN):** add "(first to drop if we fall behind)". It's a good comparison, but the scope is large (see section 5).
- **Pipeline figure:** still accurate. Its "27 low / 10 medium / 10 high" legend comes from the old data; after regenerating, re-render it for the same tower, or remove the counts.

### 4.3 Research questions
Cut to 3, as the template asks. Suggested:
1. Can a block detector trained only on simulated images correctly identify which slots of a real Jenga tower are filled?
2. Given the tower's layout, can a trained model predict each block's risk better than a simple rule of thumb, especially for medium-risk blocks, and does it still work on imperfect towers?
3. Does splitting the job into "find the blocks" and "judge the risk" work better on real photos than one end-to-end CNN?

(Look-ahead moves to a stretch goal in 4.2.)

### 5. ML techniques
- **Baselines** bullet: "a physics formula and direct simulation". Say which exist: the rule of thumb and the lookup table are built; the physics formula and "simulate the detected layout" are planned. Add them to Milestone 1 if you keep them.
- **Rating each block's risk:** add one line on class imbalance and noise: "We'll use class weights and macro-F1 so the model can't score well by mostly predicting low, and also train on the raw tilt drop, because labels near the 0.3° threshold are noisy."
- **How we'll measure it:** add "We report results separately for tidy and imperfect towers."
- **Look-ahead and RL bullets:** mark both as stretch goals, to match 4.2.

### 6. Research done
- Add **Zwick (2002), *Jenga*** (summarised in Demaine's survey *Playing Games with Algorithms*, arxiv.org/abs/cs/0106019). It uses the same stability rule as our rule of thumb and solves the idealised game exactly. That supports the "rule of thumb vs learning" framing.
- Optional: the MuJoCo time-step and solver documentation, since it informed the drift fix.

### 7. Data sources
- **Row 1:** see section 2 above.
- **Row 3 (real removal trials):** "about 50 to 100 removals" is small for checking three classes. Say what you'll measure: "agreement between simulated and real outcomes for high vs not-high", which is achievable with 50–100.
- **Row 5 (real block measurements):** add *friction*: "heights, widths and masses of 20 blocks, and wood-on-wood friction by tilting one block on another until it slides".
- **Row 6 (bot-generated towers):** not built. Mark it "(planned)" and say what it fixes: today gaps are knocked out at random, while real players choose; towers from several bots make the gaps more realistic.
- **Row 7 (ViSTR-Bench):** check the source exists and has the Jenga clips you describe before citing it. If you can't find a link, remove the row.

### 8. Initial EDA
Replace the whole section with the notebook's results:
- **Link:** `notebooks/DAP_IEDA_Jenga.ipynb` in the repository.
- **Key numbers [from notebook]:** towers, labelled blocks, photos, rule-of-thumb agreement, baseline macro-F1, baseline medium F1.
- **Add these findings, which mentors will value:**
  - *Simulator drift found and fixed.* Standing towers slowly slid apart, which inflated medium labels. Halving the time step fixed it, and a test guards it.
  - *Labels have measured noise.* Repeat measurements vary by about 0.1°, so medium starts at 3× that.
  - *A consistency check caught a labelling bug.* Blocks exactly on the 0.3° threshold were labelled low by a rounding error (1,026 blocks in the old data). Fixed and tested.
  - *Imperfect towers.* Risk mix for tidy vs imperfect **[from notebook, 7c]**.
  - *Leakage identified.* `outcome`, `max_displacement` and `max_tilt_deg` are measured after the removal, so they're never model inputs (notebook section 3 and 8).
  - *Image quality.* Brightness, contrast, sharpness and tower coverage for every photo **[from notebook, 6e]**: no blank or saturated photos, but renders are sharper and backgrounds plainer than real photos, which is the sim-to-real gap.

### 9. Milestones
- **Milestone 1:** add "generate the new 500-tower dataset (imperfect towers, drift fixed)", "measure our real Jenga set", and, if kept, "build the physics-formula baseline". Change the baseline target "macro-F1 of 0.76" to **[from notebook, section 8]**.
- **Milestone 2:** the medium target "better than 0.46" becomes **[from notebook]**.
- **Milestone 3:** fine.

### 10. Risks
Add these rows; they show the "researched risks" mentors look for:

| Risk | Impact | Mitigation |
|---|---|---|
| The simulator drifts or mislabels (both happened) | Medium labels quietly wrong | Found by consistency checks; fixed by a smaller time step and a rounding fix; automated tests now catch both. The EDA notebook re-runs all checks on every new dataset. |
| Labels near the medium threshold are noisy | Some medium/low labels are coin flips | Threshold set at 3× measured noise; also train on the raw tilt drop; report high vs not-high as the robust headline result. |
| Imperfect towers hide information from the slot layout | Same layout, different risk | Report tidy and imperfect results separately; test photo measurements (lean, sticking-out blocks) as extra features. |
| Exams from Sem 2 Week 9 | No time to fix late problems | All experiments frozen by Week 6; after that only write-up and rehearsal. |

Also fix: **"Generation bias: RL agents might be biased toward a certain path"**. That's about the bot-generated towers, not RL. Suggested: "Generation bias: towers made by one random process may not look like real games. Mitigation: towers from several bot players, results per bot, and a frozen independent test set."

### 11. Plan
- **Winter W1:** "freeze test set" depends on the new data, so add "generate the 500-tower dataset (overnight, both laptops)" before it.
- **Team notes:** add exam weeks (Sem 2 Week 9 onwards) and availability.
- **Typos:** "Repositary" → "Repository"; "Sem 2 - Weeks 6" (and 7–11) → "Week"; "Write-up," has a stray comma.
- **Sem 2 Week 8** says only "Write-up". Merge it with Week 9 or add "final evaluation" there.

---

## 5. Scope: say what gets cut first

The plan now includes a detector, a multi-task classifier (possibly a graph neural network), two new baselines, an end-to-end CNN, look-ahead, bot towers, simulator calibration, real removal trials, RL and an external benchmark, all before Week 6, by three Year 1 students. Mentors will notice. Add a priority line under section 9 or 10:

> **Must:** detector, risk classifier, baselines, real photos. **Should:** real removal trials, end-to-end CNN comparison. **Could:** look-ahead, bot towers, graph neural network, RL, ViSTR-Bench. Anything not working by Sem 2 Week 6 becomes future work, starting from the bottom of this list.

---

## 6. Charts to replace (PDF → notebook)

After running the notebook on the new data, swap each PDF chart for its replacement in `docs/eda_v2/`:

| PDF chart (page) | Replace with |
|---|---|
| Six photos of the same tower (p. 8) | Re-render with `tools/eda_figures.py` on the new data, or use `6f_examples.png` (tidy and imperfect towers, with mask and risk map) |
| How the blocks are labelled (p. 9) | `6b_block_variables.png` (risk_level panel), or re-render the single bar chart with `tools/eda_figures.py` |
| Lower blocks are more often medium risk (p. 10) | `7a_risk_height_position.png` (left) |
| Middle blocks are either safe or deadly (p. 11) | `7a_risk_height_position.png` (right) |
| Gappier towers have more dangerous blocks (p. 12) | `7b_gaps.png` (left) |
| How much does one removal weaken the tower? (p. 13) | `6b_block_variables.png` (margin_drop panel) |
| The baseline to beat (p. 14) | Table in notebook section 8 |
| **New:** tidy vs imperfect | `7c_tidy_vs_imperfect.png` |
| **New:** what's left in the layer, and rule vs physics | `7d_layer_and_rule.png` |
| **New:** image quality | `6e_image_quality.png`, `6e_pixel_intensity.png` |

The proposal's single-chart style (big title, one message per chart) comes from `tools/eda_figures.py`. For the proposal itself, re-running that script on the new data gives matching charts:

```bash
.venv/bin/python tools/eda_figures.py data/v2_train --out docs/eda_v2_proposal
```

---

## 7. More issues from an independent review

A separate reviewer read the whole PDF as a sceptical mentor and checked its claims against the repository. These are the points not already covered above, most serious first.

### Could stop a mentor picking the project
- **The real removal trials can only check high risk.** A real trial shows whether the tower falls. Medium is defined by a *simulated* tilt test, and nothing in the plan measures tilt on a real tower. So 50–100 real removals validate "high vs not high", which the rule of thumb already gets almost right. At about 14–20% medium you'd also see only 7–20 medium cases.
  **Fix:** say plainly that the real trials validate high risk. Optionally add a cheap physical tilt test (tilt a board with a phone inclinometer app until the tower falls) to check medium on a few towers.
- **The simulator removes blocks cleanly; real players pull or push them.** The dataset uses instant "delete" removals (`removal.py`). The Fazeli row admits extraction is the hardest part to simulate, but it isn't listed as a risk.
  **Fix:** add a row to section 10: "simulated removals are cleaner than a real pull, so low risk may be slightly optimistic; the real trials measure the size of this gap."
- **The milestone targets are fixed numbers from old data.** "0.76", "0.80" and "0.46" will all move after regeneration.
  **Fix:** make them relative: "beat the regenerated lookup-table baseline by at least 0.05 macro-F1, with a confidence interval."
- **Look-ahead has no defined method or cost.** "Chance of collapse over the next 3 moves" needs a rule for which moves players make. Exhaustive 3-move rollouts in physics would be about 45³ ≈ 90,000 simulations per tower.
  **Fix:** drop it to a stretch goal (as in section 4). If kept, define the move policy (e.g. random legal moves), run it on the classifier rather than the physics, and give a sampled budget.

### Major
- **The same layout can now mean different risks.** Imperfect towers mean two towers with identical slot layouts can have different labels, which caps how well a layout-only classifier can do. "Simulated and real towers look identical to it" (4.2) is no longer quite true.
  **Fix:** measure how often identical layouts disagree and report that as the ceiling, or add photo-measured lean and protrusion as optional inputs.
- **"From a single photo" vs "combine two views".** The 2.2% hidden figure counts *blocks*, but the layout needs the *gaps*, and a gap behind a block can't be seen from one side.
  **Fix:** report what share of gaps are visible in one view, and decide whether the product needs one photo or two.
- **"95% of slots right" is too easy.** Guessing "every slot filled" already scores 80–90% (most slots are filled), and 95% still allows 2–3 wrong slots per tower.
  **Fix:** measure gap recall and precision, plus "% of towers with a fully correct layout", and how much risk F1 drops when the detected layout replaces the true one.
- **Evaluation details are missing:**
  - The 30–50 real photos are used for both fine-tuning and testing, with no split. Split by physical tower (e.g. 20 for fine-tuning, the rest test only).
  - Report confidence intervals by resampling whole towers (blocks in one tower are not independent).
  - State the number of random seeds.
  - "Direct simulation" can't be a baseline on *simulated* data: it generated the labels, so it scores 100%. It only makes sense on real towers, so say that.
- **Timeline trap: freeze, then regenerate.** The plan freezes the test set in Winter W1 but calibrates the simulator and "regenerates if needed" in Sem 2 W1. That would invalidate the frozen test set, the baselines and every trained model.
  **Fix:** measure the real set and generate the final 500 towers in Winter W1–W2, *then* freeze. Also: Milestone 3 means different things in sections 9 and 11, Weeks 7–8 are nearly empty, and "final evaluation" sits in exam weeks. Move final evaluation to Weeks 7–8 and keep Weeks 9–11 as buffer.
- **Licensing and naming:**
  - Ultralytics YOLO is AGPL-3.0, which affects an "open-source" release and a hosted demo. Add a licence note, or use torchvision's Mask R-CNN (BSD licence).
  - "Jenga" is a Hasbro trademark, which matters if you publish a dataset under that name.
  - Make sure the GitHub repo is public before mentors click the link.
- **Real-data labelling effort isn't budgeted.** Block outlines for 50 photos is about 2,000 polygons. Name a tool (Roboflow, CVAT) and estimate the hours, or only outline a small subset.
- **The medium fallback undercuts the project.** "Make high vs everything else our main result": high risk is already nearly solved by a one-line rule.
  **Fix:** make the fallback "predict the tilt drop as a number (regression) and report rank correlation", which still needs ML.
- **More risks to add to section 10:**
  - the simulator changes again (it already has once);
  - real-data labelling time;
  - having a physical Jenga set and a consistent photo setup;
  - Colab GPU limits;
  - reproducibility across the two laptops (same commit and MuJoCo version).

### Minor
- **Baseline names vary:** "rule of thumb", "simple rule", "lookup table", "physics formula", "analytic statics". Pick two names, define each once, and say which are built.
- **Model names vary:** "detector", "block finder", "instance-segmentation model", "stage 1". Pick one. Also define "tilt margin" once for readers: "how far the table can be tilted before the tower falls".
- **The old margin-drop chart** (p. 12) shows bars on the old 0.15° grid, and the "medium from 0.3°" line sits at a bin edge. Many blocks shown as low at exactly 0.3° were the rounding bug. The new data fixes both, so redraw it.
- **The "r = 0.80" gaps chart** mostly confirms the generator (gap count is something it chooses). Say whether r is per tower or over the bins.
- **Re-run the "480 removals never changed outcome" check** under the new physics and imperfect towers before quoting it again.
- **EDA on held-out data:** the old EDA and baseline design used all 300 towers, including the ones later held out. The new notebook explores training data only, so say so.
- **Objectives and milestone criteria that can't be checked:** objective 1 ("every step documented"), the M3 criterion (a report, not a pass mark) and M4 ("a working demo"). Add a measurable bar to each.
- **The 4.2 box is half empty** next to the pipeline figure. Use the space for one sentence on imperfect towers, or a real-photo example once you have one.
- **Section 5 vs the plan:** tilt-margin regression is "optional" in section 5 but a core task in the Sem 2 Week 1 plan. Make them agree.

### Polish
- Stray full stop in ".Key numbers" (p. 7); leading spaces in the Winter Week 1 and 2 cells.
- "Mid-term" vs "Mid-Term".
- "Sem 2" vs the template's "Term 2".
- Name Raksha in 4.1 ("following some of the mentor's feedback").
- Your name: "S Lakshan Siva Kumar" (proposal) vs "Lakshan Sivakumar" (repo and GitHub). That's fine, but expect a mentor to ask.
- Two old pull-direction tests fail. Pull mode isn't used for the dataset, but say so up front in case a mentor runs the tests.

---

## 8. Order of work

1. Generate the dataset (both laptops, overnight).
2. Run the notebook: `JENGA_DATA=../data/v2_train` (the default).
3. Fill every **[from notebook]** value above, and swap the charts.
4. Make the template fixes (section 3) and the scope line (section 5).
5. Fill commitments, mentor reasons and team notes.
