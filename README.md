# Classifying Spacecraft Collision Risk: A Machine Learning Approach Using GRU Neural Networks

## Overview

This project provides an automated classification system for satellite collision risk using Gated Recurrent Unit (GRU) neural networks. By leveraging time-series Conjunction Data Messages (CDMs), the model helps identify high-risk collision events in low Earth orbit (LEO) and provides timely warnings to support satellite operators.

---

## Dataset

- **Source:** [ESA Collision Avoidance Challenge Dataset (2019)](https://kelvins.esa.int/collision-avoidance-challenge/data/)
- **Description:**
  - Public dataset released by the European Space Agency (ESA).
  - Contains real CDMs from close-approach events between satellites and other space objects.
  - Over 162,000 records, 103 features.
- **Note:**  
  The dataset is not included in this repository due to size restrictions.  
  **Please download `train_data.xlsx` or `train_data.csv` from the [ESA challenge page](https://kelvins.esa.int/collision-avoidance-challenge/data/) or [your provided link] and place it in the project directory.**

---

## Requirements

- Python 3.10 (recommended)
- To install required packages, run:
  ```bash
  pip install -r requirements.txt

