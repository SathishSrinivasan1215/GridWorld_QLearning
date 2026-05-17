# Q-Learning Agent: 5×5 Grid World
# Srinivasan-S.py

---

## Overview

This program trains a tabular Q-Learning agent to navigate a 5×5 grid world with obstacles and a jump shortcut. It evaluates six learning rates (α) and generates state-value heatmaps and training curve plots.

---

## Requirements

### Python Version
- Python **3.8 or higher**

### Required Libraries
Install all dependencies with:

```bash
pip install numpy matplotlib
```

| Library      | Purpose                          |
|--------------|----------------------------------|
| `numpy`      | Array operations, Q-table math   |
| `matplotlib` | Heatmap and training curve plots |
| `random`     | Built-in — no install needed     |
| `os`         | Built-in — no install needed     |

---

## How to Run

### 1. Clone or download the file
Make sure `Srinivasan-S.py` is in your working directory.

### 2. Install dependencies
```bash
pip install numpy matplotlib
```

### 3. Run the script
```bash
python Srinivasan-S.py
```

Or if your system uses `python3`:
```bash
python3 Srinivasan-S.py
```

---

## What the Program Does

1. **Trains** a Q-Learning agent for each of 6 learning rates:
   `α ∈ {1.0, 0.75, 0.5, 0.25, 0.1, 0.01}`

2. **Prints** the Q-table to the console after each training run

3. **Early stops** training if the 30-episode rolling mean reward exceeds **9.5**

4. **Saves** one state-value heatmap PNG per learning rate (6 files)

5. **Saves** one combined training curves PNG (reward + steps)

---

## Output Files

All PNG files are saved in the **same folder as the script**:

| File | Description |
|------|-------------|
| `state_values_alpha_1_0.png`    | Heatmap for α = 1.00 |
| `state_values_alpha_0_75.png`   | Heatmap for α = 0.75 |
| `state_values_alpha_0_5.png`    | Heatmap for α = 0.50 |
| `state_values_alpha_0_25.png`   | Heatmap for α = 0.25 |
| `state_values_alpha_0_1.png`    | Heatmap for α = 0.10 |
| `state_values_alpha_0_01.png`   | Heatmap for α = 0.01 |
| `training_curves.png`           | Combined reward & steps curves |
