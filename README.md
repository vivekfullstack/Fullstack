# Fullstack MLOps Repository

A collection of end-to-end Machine Learning Operations (MLOps) projects, API services, and automated data pipelines.

---

## Project Overview: `mlops_demo`

The **`mlops_demo`** subproject is a production-grade Wine Quality prediction platform featuring:

* **Training & Tracking:** Scikit-Learn RandomForest classifier tracked via **MLflow** for full lineage.
* **REST API:** High-performance **FastAPI** service (`/predict` and `/health` endpoints).
* **Drift Monitoring:** **Evidently AI** pipeline detecting feature drift (covariate shift) and prediction drift (concept shift).
* **Interactive Dashboards:** Auto-generated HTML reports (`reports/monitoring_report.html`) to trigger retraining alerts.

---

## Quickstart Guide

### 1. Prerequisite: Git Setup
Ensure Git is installed (`brew install git` on macOS, `sudo apt install git` on Linux, or from [git-scm.com](https://git-scm.com/)).

### 2. Clone & Navigate
Clone the repo and enter the project directory:

```bash
git clone https://github.com/vivekfullstack/Fullstack.git
cd Fullstack