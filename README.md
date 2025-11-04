# 📚 Teacher Allocation Optimizer

A data-driven Flask web app that helps policymakers optimize teacher allocation across Indian districts to maximize literacy improvement. Built by Shivang, an IT professional and Python learner, this project models real-world education planning using predictive analytics and interactive visualizations.

---

## 🚀 Features

- 🔄 **Strategy toggle**: Choose between `spread` (1 teacher per district) or `focused` (up to 3 per district)
- 📊 **Interactive charts**: Visualize teacher distribution and population share
- 📈 **Predictive gain modeling**: Estimate literacy improvement using tau values
- 🧮 **Backend optimization**: Allocates teachers based on tau-per-teacher scores
- 📤 **JSON output**: Transparent summary and district-level allocations

---

## 🧠 Machine Learning Model

This project uses a **tau estimation model** derived from historical education data. Tau represents the predicted literacy gain per teacher in each district. The model was trained using regression techniques on features such as:

- District-level population
- Historical literacy rates
- Teacher counts and performance metrics

Tau values are precomputed and stored in `tau_estimates.csv`, which the app uses to rank districts and allocate resources.

---

## 🛠️ Tech Stack

- **Backend**: Python, Flask, Pandas
- **Frontend**: HTML, JavaScript, Chart.js
- **Data**: CSV-based input (`tau_estimates.csv`)
- **Deployment**: Render or Railway (supports Flask + Gunicorn)

---

## 📦 Setup Instructions

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
