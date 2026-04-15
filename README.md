# 📡 MADRL LTE-U / WiFi Coexistence Simulation Platform

## 🧠 Overview

This platform is an **interactive web-based simulation tool** designed to study and analyze **spectrum efficiency and coexistence between LTE-U and WiFi networks**.

It allows users to:

* Simulate different network scenarios
* Compare coexistence strategies
* Observe system performance (throughput, fairness, packet loss)
* Visualize **Reinforcement Learning (RL) behavior over time**

---

## 🎯 Purpose

This system supports research on:

* Spectrum efficiency in unlicensed bands
* LTE-U and WiFi coexistence challenges
* Reinforcement Learning for dynamic resource allocation

---

## ⚙️ System Features

### 📊 Simulation Dashboard

* Configure:

  * Number of LTE-U agents
  * Number of WiFi access points
  * Traffic load
  * Coexistence algorithm

---

### 🤖 Reinforcement Learning (RL)

* Adaptive duty cycle control
* Reward-based learning
* Learning curve visualization

---

### 📈 Performance Metrics

After running a simulation, the system displays:

* **Throughput (Mbps)**
* **Jain’s Fairness Index**
* **Packet Loss (%)**
* **RL Learning Curve (Reward over time)**

---

## 🛠️ Requirements

Make sure you have:

* Python 3.10+
* pip
* Virtual environment (recommended)

---

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd spectrum-platform
```

---

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Run the Django Server

```bash
python manage.py runserver
```

---

### 5. Open the Application

Go to:

```
http://127.0.0.1:8000/
```

---

## 🧪 How to Use the Simulation

1. Adjust the sliders:

   * LTE-U Agents
   * WiFi APs
   * Traffic Load

2. Select an algorithm:

   * MADRL (recommended)
   * Fixed Duty Cycle
   * No Algorithm

3. Click **"Run Simulation"**

---

## 📊 Understanding Results

### 🔹 Throughput

* Measures total data successfully transmitted
* Higher is better

---

### 🔹 Fairness Index (Jain’s Index)

* Range: **0 → 1**
* 1 = Perfect fairness
* Shows balance between LTE and WiFi

---

### 🔹 Packet Loss

* Percentage of lost packets
* Lower is better

---

### 🔹 RL Learning Curve

* Shows how the RL agent improves over time
* X-axis → Simulation runs
* Y-axis → Reward

#### Interpretation:

* Early runs → unstable (learning phase)
* Later runs → stabilizing (optimal behavior)

---

## 🔌 API Endpoint (for testing)

You can test the backend using Postman:

**POST**

```
/api/simulation/run/
```

### Example Request Body:

```json
{
  "lte_agents": 3,
  "wifi_aps": 5,
  "traffic": 3,
  "algorithm": "madrl"
}
```

---

### Example Response:

```json
{
  "throughput": [14.42, 49.46, 63.88],
  "fairness": [0.769],
  "packet_loss": [2.14, 8.0],
  "rl": {
    "duty_cycle": 0.3,
    "reward": 115.19,
    "history": [121.48, 115.54, 123.53]
  }
}
```

---

## 🧠 Notes for Researchers

* RL is currently **simplified (Q-learning style)**
* Can be extended to:

  * Multi-agent RL
  * Deep Q Networks (DQN)
  * Real network datasets

---

## 🚧 Future Improvements

* Export results to CSV/PDF
* Multi-agent learning comparison
* Real-world traffic modeling
* Integration with NS-3 or MATLAB

---

## 👨‍💻 Author

Developed as part of a **Master’s research project** on:

**“Spectrum Efficiency Enhancement through LTE-U and WiFi Coexistence using Reinforcement Learning”**

---

## 📌 Summary

This platform provides:

✔ Interactive simulation
✔ Real-time visualization
✔ RL-based optimization
✔ Research-ready insights

---

## 💡 Tip

Run the simulation multiple times to see the **RL learning curve evolve** — that’s where the real insight is.

---
