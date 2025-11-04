from flask import Flask, request, jsonify, render_template
import pandas as pd

app = Flask(__name__)
TAU_PATH = "tau_estimates.csv"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/tau", methods=["GET"])
def tau():
    try:
        df = pd.read_csv(TAU_PATH)

        if "tau_per_teacher" not in df.columns:
            return jsonify({"error": "Missing 'tau_per_teacher' column in CSV."})

        df = df.sort_values("tau_per_teacher", ascending=False)
        top = df[["STATE NAME", "DISTRICT NAME", "tau_per_teacher"]].head(200)
        return jsonify(top.to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/optimize", methods=["POST"])
def optimize():
    try:
        payload = request.get_json() or {}
        budget = int(payload.get("budget", 10))
        strategy = payload.get("strategy", "spread")
        max_per_district = 1 if strategy == "spread" else 3

        df = pd.read_csv(TAU_PATH).sort_values("tau_per_teacher", ascending=False).reset_index(drop=True)

        # Ensure required columns exist
        if "total_population" not in df.columns:
            df["total_population"] = 0  # fallback if missing

        df["allocated_teachers"] = 0
        remaining = budget

        for i in range(len(df)):
            if remaining <= 0:
                break
            alloc = min(max_per_district, remaining)
            df.at[i, "allocated_teachers"] = alloc
            remaining -= alloc

        df["predicted_gain_pct"] = df["allocated_teachers"] * df["tau_per_teacher"]

        # Filter only districts with allocated teachers
        filtered = df[df["allocated_teachers"] > 0].copy()
        filtered = filtered.sort_values("predicted_gain_pct", ascending=False)

        # Drop total_population if it's 0
        filtered["total_population"] = filtered["total_population"].apply(
            lambda x: int(x) if x > 0 else None
        )

        # Build response records without total_population if it's None
        allocations = []
        for _, row in filtered.iterrows():
            record = {
                "STATE NAME": row["STATE NAME"],
                "DISTRICT NAME": row["DISTRICT NAME"],
                "allocated_teachers": int(row["allocated_teachers"]),
                "predicted_gain_pct": float(row["predicted_gain_pct"]),
                "tau_per_teacher": float(row["tau_per_teacher"])
            }
            if row["total_population"]:
                record["total_population"] = int(row["total_population"])
            allocations.append(record)

        summary = {
            "total_predicted_gain": float(df["predicted_gain_pct"].sum()),
            "budget_used": int(df["allocated_teachers"].sum())
        }

        return jsonify({"summary": summary, "allocations": allocations})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    print("✅ Flask server starting at http://127.0.0.1:5000")
    app.run(debug=True, host="127.0.0.1", port=5000)
