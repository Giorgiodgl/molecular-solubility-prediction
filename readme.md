# Molecular Solubility Prediction ($\log S$) with Explainable ML

An end-to-end cheminformatics and machine learning pipeline to predict aqueous solubility ($\text{measured log S}$) directly from SMILES strings using the ESOL (Delaney) dataset. 

This repository demonstrates **hybrid feature engineering**, **defensive hyperparameter tuning**, **Bemis-Murcko scaffold splitting**, and **SHAP (SHapley Additive exPlanations)** model interpretability.

---

## 📌 Project Overview
In early-stage drug discovery, candidate molecules must be sufficiently soluble in water to dissolve in the bloodstream and reach their biological targets. 

This project implements a complete, transparent machine learning workflow that:
1. Merges **structural Morgan fingerprints** with **explicit thermodynamic physicochemical properties**.
2. Evaluates out-of-distribution structural generalization using **Bemis-Murcko Scaffold Splitting**.
3. Audits feature importances with **SHAP TreeExplainer** to ensure alignment with fundamental chemical physics.

---

## 🛠️ Pipeline Architecture

```
Molecular SMILES ---> [ RDKit Feature Extractor ] ---> Combined Feature Matrix X (2054 cols)
                                                                 │
                                                       (2048-bit Morgan Fingerprints +
                                                        6 Physicochemical Descriptors)
                                                                 │
                                                                 ▼
[ Model Explainability (SHAP) ] <--- [ Optimized Random Forest ] <--- [ Bemis-Murcko Scaffold Split ]
```

### 1. Feature Engineering
* **Morgan Fingerprints:** 2,048-bit structural bit vectors ($radius = 2$) calculated via RDKit's `rdFingerprintGenerator`.
* **Physicochemical Properties:** 6 thermodynamic descriptors extracted directly from Delaney's dataset:
  * *Molecular Weight (MW)*
  * *Polar Surface Area (PSA)*
  * *Number of H-Bond Donors*
  * *Number of Rings*
  * *Number of Rotatable Bonds*
  * *Minimum Degree*

### 2. Scaffold-Aware Splitting
Instead of standard random shuffling (which causes data leakage when structural analogs exist across sets), the pipeline implements a **Bemis-Murcko Scaffold Split**:
* Groups molecules by core ring frameworks (`MurckoScaffoldSmiles`).
* Allocates entire molecular families to either train or test sets (~80/20 split).
* Forces the model to evaluate on **completely novel chemical scaffolds**.

### 3. Hyperparameter Optimization
Optimized via `RandomizedSearchCV` with 5-fold cross-validation to apply leaf-level regularization and prevent high-dimensional bit memorization (`n_estimators=300`, `max_depth=15`, `min_samples_split=5`, `min_samples_leaf=2`, `max_features=1.0`).

---

## 📊 Model Performance

| Dataset Split Strategy | Set | $R^2$ Score | RMSE |
| :--- | :---: | :---: | :---: |
| **Bemis-Murcko Scaffold Split** | **Train** | 0.97 | 0.37 |
| **Bemis-Murcko Scaffold Split** | **Test** | **0.40** | **1.42** |

### 💡 Domain Insight & Evaluation Takeaway
* **The Extrapolation Challenge:** While the tuned Random Forest learns the training scaffolds very well ($R^2 = 0.97$), test performance drops to $0.40$ on unseen chemical scaffolds.
* **Why This Matters:** Decision trees partition feature space based on training boundaries and cannot extrapolate to entirely novel ring structures. 
* **Key Takeaway:** Demonstrates why random train/test splits can yield artificially inflated performance metrics in drug discovery, and underlines the necessity of scaffold-aware validation when assessing AI models for virtual screening.

---

## 🧬 Model Interpretability (SHAP Analysis)

`shap.TreeExplainer` was applied to open the "black box" and evaluate feature contributions on predictions:

![SHAP Summary Plot](images/shap_summary.png)

### Chemical Consistency Audit:
1. **Molecular Weight (MW):** High MW values (red points) strongly push predictions to the **left (negative impact)**, reflecting the physical reality that larger, bulkier molecules dissolve less readily in water.
2. **Polar Surface Area (PSA):** High PSA values (red points) push predictions to the **right (positive impact)**, as polar groups form hydrogen bonds with water molecules to increase solubility.
3. **Descriptor Prominence:** Continuous thermodynamic properties dominate the top rankings over individual sparse bit fingerprints, confirming that physical properties provide a strong global predictive signal.

---

## 📁 Repository Structure

```text
molecular-solubility-prediction/
│
├── images/
│   └── shap_summary.png                  # Saved SHAP plot visualization
├── notebooks/
│   └── Molecule_Solubility_Prediction.ipynb  # End-to-end Jupyter Notebook
├── README.md                             # Project documentation
└── requirements.txt                      # Environment dependencies
```

---

## 🚀 Getting Started

### 1. Installation
Clone the repository and install required packages:

```bash
git clone https://github.com/your-username/molecular-solubility-prediction.git
cd molecular-solubility-prediction
pip install -r requirements.txt
```

### 2. Dependencies
* `python >= 3.10`
* `rdkit`
* `pandas`
* `numpy`
* `scikit-learn`
* `shap`
* `matplotlib`

---

## 🔮 Future Directions
* **Gradient Boosting Benchmark:** Evaluate XGBoost, LightGBM, and CatBoost to test leaf regularization on out-of-distribution scaffolds.
* **Graph Neural Networks (GNNs):** Implement Message Passing Neural Networks (MPNN) / AttentiveFP to learn dynamic graph representations directly from molecular structures.
* **Transfer Learning:** Fine-tune pre-trained molecular transformers like **ChemBERTa** on the ESOL dataset for enhanced scaffold generalization.