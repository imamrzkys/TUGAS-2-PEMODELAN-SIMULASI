from flask import Flask, render_template, request, send_file, redirect, url_for
import os, io, csv, time, math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

app = Flask(__name__)

PLOT_DIR = os.path.join("static", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# --------------------------
# Helpers
# --------------------------
def fmt(x, nd=6):
    return f"{x:.{nd}f}"

def now_tag():
    return str(int(time.time() * 1000))

# --------------------------
# Part 1: Rumus Tugas (DIKUNCI agar sama manual)
# --------------------------
def compute_assignment_mm2(interarrival_min: float, service_min: float):
    lam = 1.0 / interarrival_min
    mu = 1.0 / service_min

    rho = lam / (2.0 * mu)  # sesuai tugas
    denom = (mu - lam / 2.0)

    if denom <= 0:
        return None, (
            "Sistem tidak stabil untuk rumus tugas karena (μ − λ/2) ≤ 0. "
            "Percepat pelayanan atau perlambat kedatangan."
        )

    W = 1.0 / denom
    Wq = (lam ** 2) / (2.0 * mu * denom)

    steps = [
        {"title": "Input",
         "latex": rf"T_{{antar}}={interarrival_min}\ \text{{menit}},\quad T_{{layanan}}={service_min}\ \text{{menit}},\quad s=2"},
        {"title": "Laju Kedatangan (λ)",
         "latex": rf"\lambda=\frac{{1}}{{T_{{antar}}}}=\frac{{1}}{{{interarrival_min}}}={fmt(lam)}\ \text{{pelanggan/menit}}"},
        {"title": "Laju Pelayanan per Pelayan (μ)",
         "latex": rf"\mu=\frac{{1}}{{T_{{layanan}}}}=\frac{{1}}{{{service_min}}}={fmt(mu)}\ \text{{pelanggan/menit}}"},
        {"title": "Pemanfaatan (ρ)",
         "latex": rf"\rho=\frac{{\lambda}}{{2\mu}}=\frac{{{fmt(lam)}}}{{2({fmt(mu)})}}={fmt(rho)}\ (={fmt(rho*100,2)}\%)"},
        {"title": "Waktu Rata-rata Dalam Sistem (W)",
         "latex": rf"W=\frac{{1}}{{\mu-\lambda/2}}=\frac{{1}}{{{fmt(mu)}-{fmt(lam/2)}}}={fmt(W)}\ \text{{menit}}"},
        {"title": "Waktu Rata-rata Dalam Antrian (Wq)",
         "latex": rf"W_q=\frac{{\lambda^2}}{{2\mu(\mu-\lambda/2)}}=\frac{{({fmt(lam)})^2}}{{2({fmt(mu)})({fmt(mu)}-{fmt(lam/2)})}}={fmt(Wq)}\ \text{{menit}}"},
    ]

    result = {
        "lam_per_min": lam,
        "mu_per_min": mu,
        "lam_per_hr": lam * 60.0,
        "mu_per_hr": mu * 60.0,
        "rho": rho,
        "W_min": W,
        "Wq_min": Wq,
    }
    return {"result": result, "steps": steps}, None


# --------------------------
# Extra: Rumus Teori M/M/2 (Erlang-C)
# --------------------------
def compute_mm2_erlangC(interarrival_min: float, service_min: float):
    lam = 1.0 / interarrival_min
    mu = 1.0 / service_min
    s = 2

    rho = lam / (s * mu)
    if rho >= 1:
        return None, "Sistem tidak stabil untuk teori M/M/2 karena ρ ≥ 1."

    a = lam / mu  # offered load

    # P0 = [ sum_{n=0}^{s-1} a^n/n! + (a^s/s!) * 1/(1-rho) ]^-1
    sum_terms = 0.0
    for n in range(s):
        sum_terms += (a ** n) / math.factorial(n)
    tail = (a ** s) / math.factorial(s) * (1.0 / (1.0 - rho))
    P0 = 1.0 / (sum_terms + tail)

    # Erlang C: probability of waiting
    Pw = ((a ** s) / math.factorial(s)) * (1.0 / (1.0 - rho)) * P0

    # Lq, Wq, W
    Lq = Pw * (rho / (1.0 - rho))
    Wq = Lq / lam
    W = Wq + (1.0 / mu)

    theory = {
        "lam": lam,
        "mu": mu,
        "rho": rho,
        "P0": P0,
        "Pw": Pw,
        "Lq": Lq,
        "Wq": Wq,
        "W": W
    }
    return theory, None


# --------------------------
# Part 2: Simulasi Discrete-Event M/M/2
# --------------------------
def simulate_mm2(lam_per_min: float, mu_per_min: float, n_customers: int = 3000, seed: int = 42):
    rng = np.random.default_rng(seed)

    interarrival = rng.exponential(scale=1 / lam_per_min, size=n_customers)
    arrival = np.cumsum(interarrival)

    service = rng.exponential(scale=1 / mu_per_min, size=n_customers)

    server_free = np.array([0.0, 0.0])
    start = np.zeros(n_customers)
    depart = np.zeros(n_customers)
    sid = np.zeros(n_customers, dtype=int)

    for i in range(n_customers):
        s = int(np.argmin(server_free))
        sid[i] = s
        start[i] = max(arrival[i], server_free[s])
        depart[i] = start[i] + service[i]
        server_free[s] = depart[i]

    W = depart - arrival
    Wq = start - arrival

    makespan = float(depart.max())
    util = float(service.sum() / (2.0 * makespan))

    # Queue length (waiting only) over time: waiting = count(arrival<=t) - count(start<=t)
    times = np.sort(np.concatenate([arrival, start]))
    a_sorted = np.sort(arrival)
    s_sorted = np.sort(start)

    step = max(1, len(times) // 1200)
    t_list, q_list = [], []
    ai = si = 0
    for t in times[::step]:
        while ai < len(a_sorted) and a_sorted[ai] <= t:
            ai += 1
        while si < len(s_sorted) and s_sorted[si] <= t:
            si += 1
        q = ai - si
        if q < 0:
            q = 0
        t_list.append(float(t))
        q_list.append(int(q))

    return {
        "arrival": arrival,
        "service": service,
        "start": start,
        "depart": depart,
        "sid": sid,
        "W": W,
        "Wq": Wq,
        "util": util,
        "makespan": makespan,
        "t_q": np.array(t_list),
        "q_len": np.array(q_list)
    }


# --------------------------
# Plots
# --------------------------
def plot_hist_wq(Wq, outpath):
    plt.figure()
    plt.hist(Wq, bins=50)
    plt.xlabel("Wq (menit)")
    plt.ylabel("Frekuensi")
    plt.title("Histogram Wq (Simulasi)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=170)
    plt.close()

def plot_gantt(start, depart, sid, outpath, n=140):
    plt.figure()
    n = min(n, len(start))
    for i in range(n):
        y = sid[i]
        plt.plot([start[i], depart[i]], [y, y], linewidth=3)
    plt.yticks([0, 1], ["Server 1", "Server 2"])
    plt.xlabel("Waktu (menit)")
    plt.title(f"Gantt Chart Pelayanan (first {n} customers)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=170)
    plt.close()

def plot_queue_length(t_q, q_len, outpath):
    plt.figure()
    plt.plot(t_q, q_len)
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel("Waktu (menit)")
    plt.ylabel("Panjang antrian (jumlah menunggu)")
    plt.title("Panjang Antrian terhadap Waktu (Simulasi)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=170)
    plt.close()

def Wq_formula_assignment(lam, mu):
    denom = (mu - lam/2.0)
    if denom <= 0:
        return np.nan
    return (lam**2) / (2.0 * mu * denom)

def plot_3d_surface_wq(outpath):
    lams = np.linspace(0.05, 0.45, 45)
    mus = np.linspace(0.15, 0.65, 45)
    L, M = np.meshgrid(lams, mus)
    Z = np.vectorize(Wq_formula_assignment)(L, M)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(L, M, Z, rstride=1, cstride=1, linewidth=0, antialiased=True)
    ax.set_title("3D Surface Wq (Rumus Tugas) terhadap λ dan μ")
    ax.set_xlabel("λ (per menit)")
    ax.set_ylabel("μ (per menit)")
    ax.set_zlabel("Wq (menit)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=170)
    plt.close()


# --------------------------
# CSV Export
# --------------------------
def build_csv_bytes(sim_data, max_rows=200000):
    n = min(len(sim_data["arrival"]), max_rows)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["i", "arrival", "service", "start", "depart", "W", "Wq", "server"])
    for i in range(n):
        writer.writerow([
            i + 1,
            float(sim_data["arrival"][i]),
            float(sim_data["service"][i]),
            float(sim_data["start"][i]),
            float(sim_data["depart"][i]),
            float(sim_data["W"][i]),
            float(sim_data["Wq"][i]),
            int(sim_data["sid"][i]) + 1
        ])
    return output.getvalue().encode("utf-8")


# --------------------------
# Routes
# --------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/result", methods=["POST"])
def result():
    try:
        interarrival = float(request.form.get("interarrival", "").strip())
        service = float(request.form.get("service", "").strip())
        formula_mode = request.form.get("formula_mode", "tugas")  # "tugas" or "teori"
    except ValueError:
        return render_template("index.html", error="Input harus angka valid (contoh: 4 dan 3).")

    if interarrival <= 0 or service <= 0:
        return render_template("index.html", error="Input harus bernilai positif (>0).")

    payload, err = compute_assignment_mm2(interarrival, service)
    theory, terr = compute_mm2_erlangC(interarrival, service)

    if err:
        return render_template("index.html", error=err)

    return render_template(
        "result.html",
        interarrival=interarrival,
        service=service,
        formula_mode=formula_mode,
        theory=theory,
        terr=terr,
        **payload
    )

@app.route("/simulate", methods=["POST"])
def simulate():
    try:
        interarrival = float(request.form.get("interarrival", "").strip())
        service = float(request.form.get("service", "").strip())
        n_customers = int(request.form.get("n_customers", "3000"))
        seed = int(request.form.get("seed", "42"))
    except ValueError:
        return render_template("index.html", error="Input simulasi tidak valid. Pastikan angka benar.")

    if interarrival <= 0 or service <= 0:
        return render_template("index.html", error="Input harus positif (>0).")
    if n_customers < 100:
        return render_template("index.html", error="n_customers minimal 100 (misal 3000).")

    lam = 1.0 / interarrival
    mu = 1.0 / service

    sim_data = simulate_mm2(lam, mu, n_customers=n_customers, seed=seed)

    tag = now_tag()
    hist_path = os.path.join(PLOT_DIR, f"hist_wq_{tag}.png")
    gantt_path = os.path.join(PLOT_DIR, f"gantt_{tag}.png")
    qlen_path = os.path.join(PLOT_DIR, f"queue_{tag}.png")
    surf_path = os.path.join(PLOT_DIR, f"surface_{tag}.png")

    plot_hist_wq(sim_data["Wq"], hist_path)
    plot_gantt(sim_data["start"], sim_data["depart"], sim_data["sid"], gantt_path, n=140)
    plot_queue_length(sim_data["t_q"], sim_data["q_len"], qlen_path)
    plot_3d_surface_wq(surf_path)

    manual_payload, manual_err = compute_assignment_mm2(interarrival, service)
    manual_result = manual_payload["result"] if manual_payload else None

    theory, terr = compute_mm2_erlangC(interarrival, service)

    return render_template(
        "simulasi_result.html",
        interarrival=interarrival,
        service=service,
        n_customers=n_customers,
        seed=seed,
        lam=lam,
        mu=mu,
        util=sim_data["util"],
        mean_W=float(np.mean(sim_data["W"])),
        mean_Wq=float(np.mean(sim_data["Wq"])),
        p90_W=float(np.percentile(sim_data["W"], 90)),
        p90_Wq=float(np.percentile(sim_data["Wq"], 90)),
        max_W=float(np.max(sim_data["W"])),
        max_Wq=float(np.max(sim_data["Wq"])),
        hist_url="/" + hist_path.replace("\\", "/"),
        gantt_url="/" + gantt_path.replace("\\", "/"),
        qlen_url="/" + qlen_path.replace("\\", "/"),
        surf_url="/" + surf_path.replace("\\", "/"),
        manual_result=manual_result,
        theory=theory,
        terr=terr
    )

@app.route("/download_csv", methods=["GET"])
def download_csv():
    try:
        interarrival = float(request.args.get("interarrival"))
        service = float(request.args.get("service"))
        n_customers = int(request.args.get("n_customers"))
        seed = int(request.args.get("seed"))
    except:
        return redirect(url_for("index"))

    lam = 1.0 / interarrival
    mu = 1.0 / service
    sim_data = simulate_mm2(lam, mu, n_customers=n_customers, seed=seed)

    csv_bytes = build_csv_bytes(sim_data)
    return send_file(
        io.BytesIO(csv_bytes),
        mimetype="text/csv",
        as_attachment=True,
        download_name="simulasi_mm2.csv"
    )

if __name__ == "__main__":
    app.run(debug=True)
