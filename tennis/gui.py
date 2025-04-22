import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QComboBox, QPushButton,
    QVBoxLayout, QDateEdit, QMessageBox, QCompleter
)
from datetime import datetime
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from bisect import bisect_right
from collections import defaultdict

# === Wczytanie danych ===
elo_overall = pd.read_csv("elo_history_2001_2025.csv", parse_dates=["Date"])
elo_surface = pd.read_csv("elo_history_by_surface.csv", parse_dates=["Date"])
features_base = pd.read_csv("tennis_features_elo.csv")

# === Przetwarzanie historii Elo ===
def build_rating_dict(df, with_surface=False):
    rating_dict = defaultdict(list)
    for _, row in df.iterrows():
        key = (row["Player"], row["Surface"]) if with_surface else row["Player"]
        rating_dict[key].append((row["Date"], row["Rating"]))
    for key in rating_dict:
        rating_dict[key].sort()
    return rating_dict

elo_dict_overall = build_rating_dict(elo_overall)
elo_dict_surface = build_rating_dict(elo_surface, with_surface=True)

def get_latest_rating(ratings_list, match_date):
    if not ratings_list:
        return 1500
    dates = [d[0] for d in ratings_list]
    idx = bisect_right(dates, match_date)
    return ratings_list[idx - 1][1] if idx > 0 else 1500

# === Wczytanie modelu ===
class EloPredictor(torch.nn.Module):
    def __init__(self, input_size):
        super(EloPredictor, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

model = EloPredictor(input_size=10)
model.load_state_dict(torch.load("elo_model.pth", map_location=torch.device("cpu")))
model.eval()

# === Skalowanie ===
scaler = StandardScaler()
scaler.mean_ = features_base.drop("label", axis=1).mean().values
scaler.scale_ = features_base.drop("label", axis=1).std().values

# === Funkcja do tworzenia cech ===
def get_features(p1, p2, date, surface):
    p1_elo = get_latest_rating(elo_dict_overall[p1], date)
    p2_elo = get_latest_rating(elo_dict_overall[p2], date)
    p1_surf = get_latest_rating(elo_dict_surface[(p1, surface)], date)
    p2_surf = get_latest_rating(elo_dict_surface[(p2, surface)], date)
    elo_diff = p1_elo - p2_elo
    surf_elo_diff = p1_surf - p2_surf
    surface_encoding = {
        "Clay": [1, 0, 0, 0],
        "Hard": [0, 1, 0, 0],
        "Grass": [0, 0, 1, 0],
        "Carpet": [0, 0, 0, 1]
    }.get(surface, [0, 0, 0, 0])
    return [p1_elo, p2_elo, p1_surf, p2_surf, elo_diff, surf_elo_diff] + surface_encoding

# === GUI ===
class EloApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Przewidywanie meczu tenisowego (Elo NN)")
        layout = QVBoxLayout()

        # Autouzupełnianie nazwisk
        players = sorted(set(elo_overall["Player"].dropna().unique()))
        completer = QCompleter(players)
        completer.setCaseSensitivity(False)

        self.p1 = QLineEdit()
        self.p1.setCompleter(completer)

        self.p2 = QLineEdit()
        self.p2.setCompleter(completer)

        self.surface = QComboBox()
        self.surface.addItems(["Clay", "Hard", "Grass", "Carpet"])
        self.date = QDateEdit()
        self.date.setCalendarPopup(True)
        self.date.setDate(datetime.today())

        self.button = QPushButton("Przewiduj wynik")
        self.button.clicked.connect(self.predict)

        layout.addWidget(QLabel("Zawodnik A:")); layout.addWidget(self.p1)
        layout.addWidget(QLabel("Zawodnik B:")); layout.addWidget(self.p2)
        layout.addWidget(QLabel("Nawierzchnia:")); layout.addWidget(self.surface)
        layout.addWidget(QLabel("Data meczu:")); layout.addWidget(self.date)
        layout.addWidget(self.button)

        self.setLayout(layout)

    def predict(self):
        a = self.p1.text().strip()
        b = self.p2.text().strip()
        surface = self.surface.currentText()
        date = pd.Timestamp(self.date.date().toPyDate())

        try:
            x_ab = torch.tensor(scaler.transform([get_features(a, b, date, surface)]), dtype=torch.float32)
            x_ba = torch.tensor(scaler.transform([get_features(b, a, date, surface)]), dtype=torch.float32)

            prob_ab = model(x_ab).item()
            prob_ba = model(x_ba).item()

            total = prob_ab + prob_ba
            prob_A = prob_ab / total
            prob_B = prob_ba / total
        except Exception as e:
            QMessageBox.warning(self, "Błąd", f"Nie udało się obliczyć wyniku.\n{str(e)}")
            return

        msg = f"""
⚔️ {a} vs {b} – nawierzchnia: {surface}

🟢 Szansa na wygraną {a}: {prob_A * 100:.2f}%
🔴 Szansa na wygraną {b}: {prob_B * 100:.2f}%
        """
        QMessageBox.information(self, "Wynik meczu", msg.strip())

# === Uruchomienie ===
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = EloApp()
    win.show()
    sys.exit(app.exec_())
