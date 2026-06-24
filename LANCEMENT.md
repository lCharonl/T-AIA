# Guide de lancement — Taxi Driver (RL Platform)

Ce guide explique pas à pas comment installer et lancer le projet, sur
**Windows (PowerShell)** comme sur **Linux / macOS (bash)**.

> Stack : Python 3.10+ · FastAPI · WebSocket · SQLite · NumPy · PyTorch (CPU) · Gymnasium 1.0

---

## 1. Prérequis

- **Python 3.10 ou supérieur** — vérifier avec :

  ```bash
  python --version
  ```

- **pip** à jour (recommandé) :

  ```bash
  python -m pip install --upgrade pip
  ```

---

## 2. Créer et activer un environnement virtuel

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

> Si PowerShell bloque l'activation avec une erreur d'« execution policy »,
> lance d'abord :
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
> ```
> puis relance la commande d'activation.

### Linux / macOS (bash)

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Une fois activé, le prompt affiche `(.venv)` au début de la ligne.

---

## 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

> ⏳ **Note** : `torch` (PyTorch, nécessaire pour le Deep Q-Learning) est
> volumineux et son installation peut prendre plusieurs minutes. Si tu n'as
> pas besoin du DQN, tu peux installer le reste manuellement et ignorer
> `torch` / `triton`.

---

## 4. Lancer la plateforme web (mode principal)

```bash
python run_server.py
```

Puis ouvre **<http://localhost:8000>** dans le navigateur.
Le badge **`API en ligne`** doit apparaître en bas du rail de navigation.

### Variantes

```bash
# Choisir un port + activer le rechargement auto (dev)
python run_server.py --port 8765 --reload

# Lancer directement via uvicorn
uvicorn server.main:app --host 0.0.0.0 --port 8000
```

- Documentation interactive de l'API : **<http://localhost:8000/docs>**
- Pour arrêter le serveur : `Ctrl + C` dans le terminal.

---

## 5. Alternative — CLI (sans interface web)

```bash
# Comparaison rapide des 3 algos tabulaires
python main.py --compare --train 2000 --test 100 --mode time-limited

# Q-Learning seul, paramètres optimisés
python main.py --algo q-learning --train 2000 --test 100 --mode time-limited

# Deep Q-Learning
python main.py --algo dqn --train 4000 --test 100 --mode time-limited
```

---

## 6. Alternative — ancien dashboard Streamlit (backup)

```bash
streamlit run dashboard/app.py
```

> L'ancien dashboard Streamlit reste fonctionnel, mais la plateforme web
> FastAPI est désormais la référence.

---

## 7. Dépannage rapide

| Problème | Solution |
|---|---|
| `python` introuvable | Essayer `python3` (Linux/macOS) ou réinstaller Python en cochant « Add to PATH ». |
| Activation venv bloquée (PowerShell) | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` puis réactiver. |
| Le port 8000 est déjà utilisé | Lancer avec `--port 8765` (ou un autre port libre). |
| Installation de `torch` trop longue/échoue | Installer les autres dépendances et sauter le DQN, ou voir <https://pytorch.org> pour la commande adaptée à ta machine. |
| Repartir d'un historique vierge | Supprimer `server/history.db` ou utiliser le bouton *Vider l'historique* dans l'app. |

---

Pour plus de détails sur l'architecture, les algorithmes et les chiffres de
référence, voir [README.md](README.md) et [RAPPORT.md](RAPPORT.md).
