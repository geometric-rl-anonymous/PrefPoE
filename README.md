------

# Learning Where to Explore: Advantage-Guided Preference-of-Experts for Policy Optimization(PrefPoE)

This repository contains the evaluation script and pretrained model for the **HalfCheetah-v4** environment using the **PrefPoE** algorithm, as described in our upcoming ICLR submission.

## 📂 Repository Structure

```
.
├── vis.py                # Evaluation script for PrefPoE agent
├──PrefPoE.cleanrl_model      	   # Directory containing pretrained weights
├── requirements.txt               # Required Python dependencies
└── README.md                      # This file
```

## 🚀 How to Run the Evaluation

Make sure you have installed all required dependencies:

```bash
pip install -r requirements.txt
```

Then run the evaluation:

```bash
python vis.py
```

You will see output like the following:

```
Starting evaluation...
Episode 1/10: True Return=5307.90, Normalized Sum=40.13, Length=1000
Episode 2/10: True Return=6760.24, Normalized Sum=32.68, Length=1000
...
Evaluation completed!
True Episode Returns - Mean: 5319.70 ± 1957.73
...
📊 MULTI-SEED EVALUATION SUMMARY
🎯 Final Performance (across 10 seeds):
   Mean: 5616.48 ± 305.00
   Range: [5107.07, 6125.77]
```

All numerical results and plots will be saved automatically:

```
📊 数据已保存到: baseline_evaluation_data.npz  
📊 Multi-seed plots saved to: ./multi_seed_evaluation_<timestamp>.png
```

## 📈 Evaluation Results Summary

| Metric                      | Value                |
| --------------------------- | -------------------- |
| 🔁 Seeds                     | 10                   |
| 📊 Final Return (Mean ± Std) | **5616.48 ± 305.00** |
| 🧭 Return Range              | [5107.07, 6125.77]   |
| ✅ Episode Length            | 1000                 |

## 📦 Pretrained Model

The pretrained policy used for evaluation is provided under:

```
PrefPoE.cleanrl_model
```

This model corresponds to the **final policy after preference-guided training**, ready to evaluate or fine-tune.

## 🧪 Environment

- Environment: `HalfCheetah-v4`
- Framework: PyTorch + Gymnasium
- Seeds: [0, 1, ..., 9]
- Evaluation length: 10 episodes per seed

## 📜 License

This code is released under the MIT License. See the `LICENSE` file for details.

## 📫 Contact

If you have questions regarding this repository, please contact us **via the anonymous ICLR OpenReview submission platform**.

------

