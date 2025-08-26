# Neural Networks — Applied Projects (5-in-1)

This repository bundles five applied neural-network projects. Each project is self-contained (runnable in Google Colab or locally) and framed as a portfolio artifact. Source write‑ups are in **Project_Papers/**.

---

## 1) Spherical Projection of 3D Gaussian Data (with Hyperparameter Tuning)

**Objective.** Learn a mapping from 3D Gaussian samples to their unit‑sphere projection.  
**Approach.** Feed‑forward net (3→20→20→3, ReLU), **MSE** loss, **Adam** optimizer; 60/20/20 split. Hyperparameter search (batch size, LR, width/depth, epochs, optimizer) guided the final choice.  
**Result (held‑out).** Test MSE ≈ 0.000871; fast convergence with Adam.  
**Notebooks.**
- `Spherical_projection_gaussian.ipynb`
- `Search_for_best_hyperparameters for Spherical Projection.ipynb`

---

## 2) Generative Modeling: Gaussian ↔ Uniform & 1D → 2D Gaussian (VAE)

**Scope.**
- **f1:** 2D Gaussian → 2D Uniform  
- **f2:** 2D Uniform → 2D Gaussian (inverse)  
- **f3:** 1D Uniform → 2D Gaussian via a **VAE‑style** head (reparameterization)

**Approach.** f1/f2 trained with **MMD** as the distribution‑matching objective; compared **L1** vs **L2** regularization. f3 used a VAE formulation (mean/log‑var + sampling) to succeed on the 1D→2D task.  
**Notebook.** `GaussainUniform.ipynb`

---

## 3) Shape Generation & Latent Information (Autoencoders vs VAE)

**Objective.** Generate 28×28 images with a single geometric shape at random locations; compare **VAE** vs **basic AE** on reconstruction and generation.  
**Approach.** Conv VAE (ELBO) vs AE with Gaussian noise injected in the latent; metrics: **MSE**, **SSIM**, latent‑space interpolation and t‑SNE.  
**Results (summary).** AE slightly edges reconstruction (e.g., SSIM ~0.9934 vs 0.9931), while VAE excels at sampling and smooth latent interpolations; information-through-latent estimated (bits).  
**Notebook.** `Shape Generation and Latent Information in Autoencoders.ipynb`

---

## 4) Tracking Random Walks with Sensor Networks & Graph Neural Networks

**Objective.** From noisy sensor activations, predict which sensors were actually visited by a moving target (tiger).  
**Approach.** **Graph Attention Network (GAT)**: 2‑D node features → 20‑hidden → 1‑logit, 4 heads, **BCELoss**, **Adam**; evaluate effects of detection radius **d**, false‑positive probability **q**, and communication range **u**; compare **256** vs **512** sensors.  
**Findings (summary).** Naïve training overfits (~92%); with 80/20 split and tuning, ~80% accuracy; accuracy ↑ with **d**, ↓ with **q**, mild ↑ with **u**; 512‑sensor setup generally stronger.  
**Notebook.** `Tracking Random Walks with Sensor Networks and Graph Neural Networks.ipynb` (use your exact filename)

---

## 5) Distribution Mapping Using Diffusion Processes: SDEs & ODEs

**Objective.** Compare **SDE** (Euler‑Maruyama) vs **ODE** (deterministic velocity field) for mapping distributions using two‑pixel “image” data.  
**Tasks.** Gaussian → Dog (SDE and ODE) and Cat → Dog (ODE).  
**Metrics.** MSE, Wasserstein distance, Histogram Intersection, KL divergence, processing time.  
**Findings (summary).** ODE consistently outperforms SDE on distributional metrics (e.g., lower Wasserstein, higher Histogram Intersection) and runs faster; Cat→Dog ODE achieves the best overall scores.  
**Notebook/Paper.** See `Project_Papers/` and add notebook path if present.

---

## Repository Layout

```
.
├─ Project_Papers/
│  ├─ Project_01.pdf
│  ├─ Project_02.pdf
|  ├─ Project_03.pdf
|  ├─ Project_04.pdf
|  ├─ Project_05.pdf
├─ GaussainUniform.ipynb
├─ Search_for_best_hyperparameters for Spherical Projection.ipynb
├─ Shape Generation and Latent Information in Autoencoders.ipynb
├─ Spherical_projection_gaussian.ipynb
├─ Tracking Random Walks with Sensor Networks and GNNs.ipynb
├─ readme.md   # this file
```

---

## Quickstart

**Run in Colab:** open any notebook and run all cells.  
**Run locally:**
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\Activate
pip install -r requirements.txt
jupyter lab   # or: jupyter notebook
```

**Suggested `requirements.txt`:**
```
numpy
pandas
scikit-learn
torch
matplotlib
tqdm
# only for the GNN project:
torch-geometric
```

---

## Notes

- All code is original and executed in Google Colab.  
- Hyperparameter search for the spherical‑projection task is part of **Project 1** (separate notebook provided).  

