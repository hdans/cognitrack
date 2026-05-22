"""
Hierarchical Fuzzy Logic (Mamdani) for Student Engagement Detection.
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# Define universe of discourse
def build_universes():
    ear_universe    = np.arange(0.0, 0.51, 0.01)   # 0.00 - 0.50 (EAR)
    pose_universe   = np.arange(0, 91, 1)           # 0 - 90 derajat
    emosi_universe  = np.arange(0, 10.1, 0.1)       # 0 - 10 (proxy kelas)
    engage_universe = np.arange(0, 101, 1)          # 0 - 100 (skor output)

    return ear_universe, pose_universe, emosi_universe, engage_universe


# Initialize antecedent and consequent
def build_fuzzy_variables():
    ear_univ, pose_univ, emosi_univ, engage_univ = build_universes()

    # --- Antecedent 1: EAR (Eye Aspect Ratio) ---
    ear = ctrl.Antecedent(ear_univ, "EAR")
    ear["Mengantuk"]  = fuzz.trapmf(ear.universe, [0.00, 0.00, 0.10, 0.15])
    ear["Lelah"]      = fuzz.trimf(ear.universe,  [0.10, 0.20, 0.28])
    ear["Waspada"]    = fuzz.trapmf(ear.universe, [0.25, 0.30, 0.50, 0.50])

    # --- Antecedent 2: Pose (Deviasi Sudut Kepala) ---
    pose = ctrl.Antecedent(pose_univ, "Pose")
    pose["Fokus"]      = fuzz.trapmf(pose.universe, [0,  0, 10, 20])
    pose["Menghindar"] = fuzz.trapmf(pose.universe, [20, 30, 90, 90])

    # --- Antecedent 3: Emosi (proxy output CNN) ---
    emosi = ctrl.Antecedent(emosi_univ, "Emosi")
    emosi["Negatif"]  = fuzz.trimf(emosi.universe, [0.0, 0.0, 3.5])
    emosi["Netral"]   = fuzz.trimf(emosi.universe, [2.5, 5.0, 7.5])
    emosi["Positif"]  = fuzz.trimf(emosi.universe, [6.5, 10.0, 10.0])

    # --- Consequent: Engagement Score ---
    engagement = ctrl.Consequent(engage_univ, "Engagement", defuzzify_method="centroid")
    engagement["Rendah"]  = fuzz.trapmf(engagement.universe, [0,   0,  20, 35])
    engagement["Sedang"]  = fuzz.trimf(engagement.universe,  [25, 50,  75])
    engagement["Tinggi"]  = fuzz.trapmf(engagement.universe, [65, 80, 100, 100])

    return ear, pose, emosi, engagement


# Build rule base
def build_rules(ear, pose, emosi, engagement):
    rule1 = ctrl.Rule(
        antecedent=(ear["Waspada"] & pose["Fokus"] & emosi["Positif"]),
        consequent=engagement["Tinggi"],
        label="R1: Waspada+Fokus+Positif -> Tinggi"
    )
    rule2 = ctrl.Rule(
        antecedent=(ear["Waspada"] & pose["Fokus"] & emosi["Netral"]),
        consequent=engagement["Tinggi"],
        label="R2: Waspada+Fokus+Netral -> Tinggi"
    )
    rule3 = ctrl.Rule(
        antecedent=(ear["Waspada"] & pose["Fokus"] & emosi["Negatif"]),
        consequent=engagement["Sedang"],
        label="R3: Waspada+Fokus+Negatif -> Sedang (hadir tapi emosi negatif)"
    )
    rule4 = ctrl.Rule(
        antecedent=ear["Mengantuk"],
        consequent=engagement["Rendah"],
        label="R4: Mengantuk -> Rendah (tanpa peduli pose/emosi)"
    )
    rule5 = ctrl.Rule(
        antecedent=pose["Menghindar"],
        consequent=engagement["Rendah"],
        label="R5: Menghindar -> Rendah (tidak menatap layar)"
    )
    rule6 = ctrl.Rule(
        antecedent=(ear["Waspada"] & pose["Fokus"] & emosi["Negatif"]),
        consequent=engagement["Sedang"],
        label="R6: Waspada+Fokus+Negatif -> Sedang (hadir tapi emosi negatif)"
    )
    rule7 = ctrl.Rule(
        antecedent=(ear["Lelah"] & pose["Fokus"] & emosi["Netral"]),
        consequent=engagement["Sedang"],
        label="R7: Lelah+Fokus+Netral -> Sedang"
    )
    rule8 = ctrl.Rule(
        antecedent=(ear["Lelah"] & pose["Menghindar"]),
        consequent=engagement["Rendah"],
        label="R8: Lelah+Menghindar -> Rendah"
    )
    rule9 = ctrl.Rule(
        antecedent=(ear["Waspada"] & pose["Menghindar"] & emosi["Netral"]),
        consequent=engagement["Sedang"],
        label="R9: Waspada+Menghindar+Netral -> Sedang (sebentar lihat samping)"
    )

    rules = [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9]
    print(f"\n=== Rule Base ({len(rules)} aturan) ===")
    for r in rules:
        print(f"  {r.label}")
    return rules


# Build control system
def build_control_system(rules):
    system     = ctrl.ControlSystem(rules)
    simulation = ctrl.ControlSystemSimulation(system)
    print("\n=== Control System berhasil dibangun ===")
    return system, simulation


# Class mapping

# Mapping nama kelas CNN ke nilai crisp pada domain Emosi [0, 10]
CNN_CLASS_TO_EMOSI = {
    "Negatif": 0.0,
    "Netral":  5.0,
    "Positif": 10.0,
}

EMOTION_CLASS_ORDER = ("Negatif", "Netral", "Positif")

def class_to_emosi_value(class_name: str) -> float:
    val = CNN_CLASS_TO_EMOSI.get(class_name)
    if val is None:
        raise ValueError(f"Nama kelas tidak dikenal: '{class_name}'. "
                         f"Pilih dari: {list(CNN_CLASS_TO_EMOSI.keys())}")
    return val


def emotion_input_to_value(emotion_input) -> float:
    """Convert class name, crisp scalar, or probability distribution to a crisp Emosi value."""
    if isinstance(emotion_input, (int, float, np.floating, np.integer)):
        return float(np.clip(emotion_input, 0.0, 10.0))

    if isinstance(emotion_input, str):
        return class_to_emosi_value(emotion_input)

    if isinstance(emotion_input, dict):
        probs = np.array([float(emotion_input.get(cls, 0.0)) for cls in EMOTION_CLASS_ORDER], dtype=float)
    else:
        probs = np.asarray(emotion_input, dtype=float).reshape(-1)

    if probs.size != len(EMOTION_CLASS_ORDER):
        raise ValueError(
            f"Soft emotion input harus punya {len(EMOTION_CLASS_ORDER)} nilai untuk {EMOTION_CLASS_ORDER}."
        )

    probs = np.clip(probs, 0.0, None)
    total = probs.sum()
    if total <= 1e-8:
        probs = np.array([1.0 / len(EMOTION_CLASS_ORDER)] * len(EMOTION_CLASS_ORDER), dtype=float)
    else:
        probs = probs / total

    crisp_values = np.array([CNN_CLASS_TO_EMOSI[cls] for cls in EMOTION_CLASS_ORDER], dtype=float)
    return float(np.dot(probs, crisp_values))


# Main inference function
def compute_engagement(simulation, ear_val: float, pose_val: float,
                        emosi_val, verbose: bool = True) -> float:
    # Kliping nilai agar tetap dalam domain
    ear_val   = float(np.clip(ear_val,   0.00, 0.50))
    pose_val  = float(np.clip(pose_val,  0,    90))
    emosi_val = emotion_input_to_value(emosi_val)

    simulation.input["EAR"]   = ear_val
    simulation.input["Pose"]  = pose_val
    simulation.input["Emosi"] = emosi_val

    simulation.compute()

    score = simulation.output["Engagement"]

    if verbose:
        print(f"\n--- Inferensi Fuzzy ---")
        print(f"  Input EAR   = {ear_val:.3f}")
        print(f"  Input Pose  = {pose_val:.1f} derajat")
        print(f"  Input Emosi = {emosi_val:.1f}")
        print(f"  -> Engagement Score = {score:.2f} / 100")

    return score


# Membership functions visualization
def plot_membership_functions(ear, pose, emosi, engagement):
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig)
    fig.suptitle("Membership Functions - Fuzzy Engagement Detection",
                 fontsize=15, fontweight="bold")

    # Warna konsisten per himpunan
    COLOR_MAP = {
        "Mengantuk": "#e74c3c", "Lelah": "#e67e22", "Waspada": "#2ecc71",
        "Fokus": "#3498db",     "Menghindar": "#e74c3c",
        "Negatif": "#e74c3c",   "Netral": "#f1c40f",    "Positif": "#2ecc71",
        "Rendah": "#e74c3c",    "Sedang": "#f39c12", "Tinggi": "#2ecc71"
    }

    variables = [
        (ear,       "EAR (Eye Aspect Ratio)",         gs[0, 0]),
        (pose,      "Pose (Deviasi Sudut, derajat)",  gs[0, 1]),
        (emosi,     "Emosi (Proxy CNN)",               gs[1, 0]),
        (engagement,"Engagement Score",                gs[1, 1]),
    ]

    for var, title, grid_pos in variables:
        ax = fig.add_subplot(grid_pos)
        for label, mf in var.terms.items():
            color = COLOR_MAP.get(label, "steelblue")
            ax.plot(var.universe, mf.mf, label=label, color=color, linewidth=2)
            ax.fill_between(var.universe, 0, mf.mf, color=color, alpha=0.12)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Nilai")
        ax.set_ylabel("Derajat Keanggotaan (mu)")
        ax.set_ylim(-0.05, 1.15)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig("membership_functions.png", dpi=150)
    plt.show()
    print("\nMembership functions disimpan: 'membership_functions.png'")


# Demo simulation
def run_simulation_demo(simulation):
    print("\n" + "=" * 60)
    print("  DEMO SIMULASI - Beberapa Skenario")
    print("=" * 60)

    scenarios = [
        {
            "nama":    "Siswa Aktif & Perhatian",
            "ear":     0.35,    # Waspada
            "pose":    5.0,     # Fokus lurus ke depan
            "emosi":   "Positif",
            "expected": "Tinggi (~80-95)"
        },
        {
            "nama":    "Siswa Netral tapi Perhatian",
            "ear":     0.32,    # Waspada
            "pose":    8.0,     # Fokus
            "emosi":   "Netral",
            "expected": "Sedang (~40-65)"
        },
        {
            "nama":    "Siswa Mengantuk",
            "ear":     0.08,    # Mengantuk (mata hampir tertutup)
            "pose":    10.0,    # Pose masih oke
            "emosi":   "Netral",
            "expected": "Rendah (~10-30)"
        },
        {
            "nama":    "Siswa Menghindar (Distraksi)",
            "ear":     0.30,    # Mata terbuka
            "pose":    45.0,    # Kepala menoleh jauh
            "emosi":   "Netral",
            "expected": "Rendah (~10-30)"
        },
        {
            "nama":    "Siswa Lelah tapi Netral",
            "ear":     0.22,    # Lelah
            "pose":    12.0,    # Masih fokus
            "emosi":   "Netral",
            "expected": "Sedang (~35-55)"
        },
        {
            "nama":    "Siswa Aktif tapi Emosi Negatif",
            "ear":     0.35,    # Waspada
            "pose":    7.0,     # Fokus
            "emosi":   "Negatif",
            "expected": "Sedang (~40-60)"
        },
    ]

    results = []
    print(f"\n{'No':<3} {'Skenario':<35} {'EAR':<6} {'Pose':>5} {'Emosi':>8} {'Score':>8} {'Expected'}")
    print("-" * 85)

    for i, sc in enumerate(scenarios, 1):
        emosi_val = class_to_emosi_value(sc["emosi"])
        score     = compute_engagement(
            simulation,
            ear_val=sc["ear"],
            pose_val=sc["pose"],
            emosi_val=emosi_val,
            verbose=False
        )
        results.append(score)
        print(
            f"{i:<3} {sc['nama']:<35} {sc['ear']:<6.2f} "
            f"{sc['pose']:>5.1f} {sc['emosi']:>8} {score:>8.2f}   {sc['expected']}"
        )

    print("\nOK - Semua skenario berhasil dikomputasi.")
    return results


# Main Execution

def main():
    print("=" * 60)
    print("  FASE 2: HIERARCHICAL FUZZY LOGIC - Mamdani FIS")
    print("  Sistem Deteksi Engagement Siswa Pembelajaran Jarak Jauh")
    print("=" * 60)

    # 1. Bangun variabel fuzzy
    ear, pose, emosi, engagement = build_fuzzy_variables()

    # 2. Definisikan rule base
    rules = build_rules(ear, pose, emosi, engagement)

    # 3. Rakit control system
    _, simulation = build_control_system(rules)

    # 4. Visualisasi membership function
    plot_membership_functions(ear, pose, emosi, engagement)

    # 5. Demo simulasi beberapa skenario
    run_simulation_demo(simulation)

    # 6. Contoh inferensi satu nilai spesifik
    print("\n" + "=" * 60)
    print("  CONTOH INFERENSI TUNGGAL")
    print("=" * 60)

    # Skenario: Siswa waspada, sedikit menoleh, netral
    ear_input   = 0.30               # Nilai EAR dari detektor wajah
    pose_input  = 18.0               # Deviasi sudut dari head pose estimator
    emosi_input = class_to_emosi_value("Netral")  # Output dari model CNN Fase 1

    skor = compute_engagement(
        simulation,
        ear_val=ear_input,
        pose_val=pose_input,
        emosi_val=emosi_input,
        verbose=True
    )

    # Interpretasi kategoris
    if skor >= 65:
        kategori = "TINGGI - Siswa sangat terlibat dalam pembelajaran"
    elif skor >= 35:
        kategori = "SEDANG - Keterlibatan siswa moderat, perlu perhatian"
    else:
        kategori = "RENDAH - Siswa tidak aktif, perlu intervensi"

    print(f"\n  Skor Engagement  : {skor:.2f} / 100")
    print(f"  Kategori         : {kategori}")

    print("\nOK - Pipeline Fase 2 selesai.")
    print(f"  File MF plot : membership_functions.png")


if __name__ == "__main__":
    main()
