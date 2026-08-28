# LARP: A Language-Action Robotic Planner Integrating Retrieval-Augmented LLMs and Dynamic Scene Graphs for Detecting and Repairing Execution Failures in Robotics Manipulation Tasks

LARP detects and repairs missing steps in LLM-generated robot manipulation
plans. A Graph Attention Network (GAT) verifier rolls a plan forward over a
Dynamic Scene Graph, flags precondition violations (missing prerequisites,
obstructions, containment, gripper occupancy), identifies the responsible
object, and drives a deterministic repair — before the plan reaches the robot.

The system is demonstrated end-to-end: natural-language command → LLM
planning → GAT verification and repair → task-oriented grasp synthesis
(SenseTask) → execution on a real 3-DOF Delta parallel robot, with the scene
re-perceived after every action.

LLMs translate instructions into plausible action sequences, but they are
disembodied — they omit physically necessary steps, such as picking up a tool
before using it, or ignoring that a target object is blocked by something on
top of it. LARP is not another planner; it verifies and repairs plans a
planner already produced, using a learned model that stays reliable under
noisy, real-world perception where hand-coded symbolic rules degrade.

## Architecture

```
command
  │
  ▼
Strategic Layer (LLM + RAG)  ──────►  Instructional Task Graph (ITG)
                                       (action, object, target) trios
  │
  ▼
Tactical Layer (GAT verifier)  ────►  detect violation → repair trio
  │            ▲
  │            └── Dynamic Scene Graph, rebuilt after every action
  ▼
Grasping (SenseTask)  ─────────────►  task-oriented grasp point
  │
  ▼
Delta robot execution  ────────────►  motion, re-capture, re-verify
```

Planning runs once per command. Perception and verification run after every
executed action.

## Repository layout

```
LARP/
├── domain/            action/precondition definitions, world state
├── dataset/           benchmark generator, corruption/noise model, labeler
├── model/             GAT verifier, training, evaluation, ablations
├── strategic/         LLM planning agent (Claude; Gemini variant included)
├── perception/        YOLOE detection → scene graph
├── grasping/          SenseTask integration, trio → grasp routing
├── delta_robot_manager/  robot control library (hardware interface)
├── larp_production.py   closed-loop pipeline: perception → plan → verify →
│                         repair → grasp → execute → re-perceive
├── run_pipeline.py       offline/dry-run version of the pipeline (no hardware)
└── paper/                LaTeX source for the Expert Systems with Applications
```

## Setup

```bash
pip install -r requirements.txt
```

Set your LLM API key as an environment variable (never commit it):

```bash
# Windows PowerShell
setx ANTHROPIC_API_KEY "your-key-here"
```

Model weights (`yoloe-v8s-seg.pt`, SenseTask `.pth` files) are not tracked in
this repository — download them separately (see `perception/` and
`grasping/SenseTask/` for sources) and place them in the paths referenced by
each module's config.

## Running it

**Test the benchmark generator and GAT verifier (no hardware needed):**
```bash
python dataset/generate.py --noise 1.0
python model/train_gat.py
```

**Test perception + planning on a photo (no robot):**
```bash
python perception/perception.py --image path/to/table.jpg --conf 0.15
```

**Run the full pipeline logic without moving anything:**
```bash
python larp_production.py --command "make a sandwich" --image table.jpg
```

**Run on the real Delta robot** (requires the physical setup and calibration
described in `delta_robot_manager/`):
```bash
python larp_production.py --command "make a sandwich" --live
```
Every commanded motion is gated by a mandatory operator confirmation before
the robot moves.

## Benchmark

The dataset generator produces episodes with ground-truth
`(action, object, target)` plans, labeled corruptions across a four-category
failure taxonomy (obstruction, containment, occupancy, omitted prerequisite),
and a perception-noise model that independently corrupts redundant scene-graph
cues (edges and node flags) to simulate imperfect detection. Splits withhold
object classes and failure configurations to measure generalization
separately from memorization.

## Status

This repository accompanies a manuscript in preparation for *Expert Systems with Applications*. Some experiments referenced in the paper (baseline
comparisons, ablations) are being finalized; see `paper/` for the current
draft.

## Citation

```bibtex
@article{larp2026,
  title   = {LARP: A Language-Action Robotic Planner Integrating Retrieval-Augmented LLMs and Dynamic Scene Graphs for Detecting and Repairing Execution Failures in Robotics Manipulation Tasks},
  author  = {Elaheh Alizadehmanqhootae},{Zeinab Ezzati}
  journal = {Expert Systems with Applications},
  year    = {2026},
  note    = {under review}
}
```

## Acknowledgments

Physical experiments were conducted on the 3-DOF Delta parallel robot at the
Human-Robot Interaction Laboratory, University of Tehran, in collaboration
with TaarLab.


