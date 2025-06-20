# Mapping Conformational Transitions of a Lipid Bilayer on a Graph Neural Network

![GNN Architecture](./images/gnn_arc.png)


This repository contains the code and resources associated with our research on using **Graph Neural Networks (GNNs)** to predict and analyze **conformational transitions** in lipid bilayers from molecular dynamics (MD) simulations.

## 🧠 Overview

Understanding pore formation and closure in lipid bilayers is crucial for studying many biophysical processes. Traditional Molecular Dynamics (MD) simulations provide detailed insight but are computationally expensive and require effective **reaction coordinates (RCs)** for enhanced sampling.

In this study, we:

- Develop a **GNN-based framework** to predict four physically meaningful **collective variables (CVs)**.
- Train the GNN on 3D coordinates of coarse-grained beads from MD simulations.
- Use **rotational data augmentation** to enforce invariance and improve model performance.
- Analyze the **latent space** using dimensionality reduction (UMAP, t-SNE) to extract **machine learning CVs**.
- Build a pipeline for real-time prediction of CVs using a downstream **Artificial Neural Network (ANN)**.

## 📈 Key Features

- ✅ High prediction accuracy (R² > 0.9 for 3/4 CVs)
- 🔄 Rotational data augmentation improves generalization
- 🔍 Latent space visualization reveals biophysical transitions
- 🧩 Dimensionality reduction (UMAP) to extract interpretable, compressed CVs
- ⚡ Real-time CV prediction pipeline using ANN

## 🧬 Methodology

- **Input**: Coarse-grained MD trajectory data of DPPC lipid bilayers
- **Graph Construction**: Nodes = beads, Edges = spatial proximity (cutoff from RDF)
- **Model**: GNN with 3 message-passing hops and 64-dim hidden embeddings
- **Training**: Adam optimizer, MSE loss, data augmentation via random rotation
- **CVs Predicted**:
  - s(R) — path CV
  - ξ — Tolpekina's pore size parameter
  - ξmf — Mirjalili-Feig water density-based metric
  - z(R) — orthogonal path CV

## 🔄 ML CV Pipeline

1. MD frame → Graph
2. Graph → GNN → 64-dim embedding
3. Embedding → ANN → 4-dim UMAP-based CVs
4. (Optional) Predict physical CVs from UMAP CVs


## ⚙️ Requirements

- torch
- numpy
- pandas
- matplotlib
- tqdm
- scikit-learn
- scipy
- plotly
- MDAnalysis
- umap-learn
- gdown

Install dependencies:

```bash
pip install -r requirements.txt
