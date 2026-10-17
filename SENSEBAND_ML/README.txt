# SENSEBAND ML + Website

## 1. Install Python packages
Open a terminal in this folder and run:

    pip install -r requirements.txt

## 2. Start the website + ML server
Run:

    python app.py

Then open:

    http://127.0.0.1:5000

Do NOT open index.html directly with double-click. Open the Flask URL above.

## 3. How it works

The browser generates the simulated heart rate, sound, light and movement values.
Every update is sent to the Python Flask endpoint:

    POST /predict

The Python server loads `senseband_model.pkl`, runs the Decision Tree, and sends
back `Normal` or `Alert`.

## 4. Retrain after adding more data

Replace `senseband_data.csv` with your updated dataset, then run:

    python train_model.py

Then restart:

    python app.py

## Important

The current dashboard still contains the threshold input controls because they are
useful for visualizing sensor limits. The actual Normal/Alert classification is now
made by the ML model, not those threshold comparisons.
