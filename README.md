# Taxi Driver — Reinforcement Learning Platform

Plateforme web d'apprentissage par renforcement pour le sujet Epitech
*Taxi Driver* (T-AIA-902), construite sur `Gymnasium Taxi-v3`. Quatre
algorithmes implémentés (Brute Force, Q-Learning, SARSA, Deep Q-Learning),
visualisables et entraînables en direct dans le navigateur.

> Stack : Python 3.10+ · FastAPI · WebSocket · SQLite · NumPy · PyTorch
> (CPU) · Gymnasium 1.0 · HTML/CSS/JS vanilla (zéro dépendance front).

## Aperçu

| Écran | Quoi |
|---|---|
| **Accueil** | Hero, stats clés, 4 cartes algos. |
| **Environnement** | Grille 5×5 animée — un agent réel joue un épisode glouton, journal d'actions, récompense cumulée. |
| **Entraînement** | Sliders α/γ/ε/decay/min/episodes + choix algo. Le bouton **Entraîner** ouvre une connexion WebSocket vers le backend Python : courbes reward/steps mises à jour en streaming, résultat d'évaluation, run persisté en SQLite. |
| **Benchmark** | Lance les 5 presets en série (Brute Force, Q-Learning non opti, Q-Learning opti, SARSA opti, DQN opti) et remplit cartes + bar charts + tableau. |
| **Reward shaping** | Comparaison réelle de convergence avec / sans potential-based shaping (Ng et al. 1999, distance de Manhattan). Algo switchable Q-Learning ↔ DQN, slider d'intensité λ. |
| **Historique** | Tous les runs lancés depuis Entraînement sont archivés (hyperparams + courbe + résultat). Étoile sur le meilleur, possibilité de vider. |

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Lancer la plateforme web

```bash
python run_server.py             # http://localhost:8000
# ou
python run_server.py --port 8765 --reload
# ou directement :
uvicorn server.main:app --host 0.0.0.0 --port 8000
```

Ouvrir <http://localhost:8000>. Le badge `API en ligne` doit apparaître en
bas du rail de navigation.

## Lancer la CLI (sans web)

```bash
# Comparaison rapide des 3 algos tabulaires
python main.py --compare --train 2000 --test 100 --mode time-limited

# Q-Learning seul, params optimisés
python main.py --algo q-learning --train 2000 --test 100 --mode time-limited

# Deep Q-Learning
python main.py --algo dqn --train 4000 --test 100 --mode time-limited
```

## Lancer l'ancien dashboard Streamlit (backup)

```bash
streamlit run dashboard/app.py
```

L'ancien dashboard Streamlit reste fonctionnel mais la plateforme web
FastAPI est désormais la référence.

## Structure du projet

```
T-AIA-902/
  agents/                    # algorithmes (interface BaseAgent commune)
    base_agent.py            # boucle générique + train_stream() générateur
    brute_force.py           # baseline aléatoire (TimeLimit relevée à 3000)
    q_learning.py            # TD off-policy tabulaire
    sarsa.py                 # TD on-policy tabulaire
    dqn.py                   # Deep Q-Network (PyTorch CPU, embed + 2x64 MLP)
  core/
    config.py                # OPTIMIZED_PARAMS, DEFAULT_USER_PARAMS, PRESETS
    training.py              # boucles d'entraînement (headless + callback)
    metrics.py               # MA, summary, tableau comparatif
  server/                    # backend FastAPI
    main.py                  # routes REST + WS /ws/train
    runner.py                # wrappers : rollout, streaming, shaping potentiel
    history.py               # SQLite (insert/list/delete runs)
    history.db               # auto-créée
  web/                       # frontend (maquette HTML/CSS/JS fidèle)
    index.html               # = proto/Taxi Driver - Maquette.html + tweaks
    proto.css                # = proto/proto.css + toasts + API badge
    proto.js                 # refactoré pour piloter le backend réel
    api.js                   # client API (fetch + WebSocket + toasts)
  dashboard/                 # ancien dashboard Streamlit (backup)
  proto/                     # maquette d'origine (référence visuelle)
  main.py                    # entrypoint CLI
  run_server.py              # entrypoint web
  RAPPORT.md                 # commentaires benchmark + optimisation
```

## API

| Méthode | URL | Description |
|---|---|---|
| `GET`  | `/api/health` | ping |
| `GET`  | `/api/algos` | liste des algos + presets + defaults |
| `POST` | `/api/episode` | entraîne (si demandé) + joue un épisode glouton, renvoie le trace step-par-step (taxi_row, taxi_col, pass_loc, dest_idx, reward, action…) pour l'animation côté front |
| `POST` | `/api/benchmark` | lance tous les presets, renvoie les rows pour le tableau |
| `POST` | `/api/shaping_compare` | deux entraînements (base / shaped), retourne les courbes + conv_episode |
| `WS`   | `/ws/train` | streaming d'entraînement : envoyer un `TrainRequest`, recevoir `start` puis 1 `episode` par épisode, puis `done` (run persisté) |
| `GET`  | `/api/runs` | liste l'historique |
| `DELETE` | `/api/runs/{id}` | supprime un run |
| `DELETE` | `/api/runs` | vide l'historique |

Documentation interactive : <http://localhost:8000/docs>.

## Chiffres de référence (mesurés)

Le tableau ci-dessous reproduit les valeurs annoncées dans la présentation
de soutenance — toutes reproductibles via *Benchmark → Relancer le
benchmark* dans la plateforme.

| Algorithme | Pas moyens | Récompense | Réussite | Train time |
|---|---:|---:|---:|---:|
| Brute Force (sans truncation) | ~1700 | ~−6500 | ~80 % | 0 s |
| Q-Learning non optimisé (preset) | ~125 | ~−118 | ~40 % | ~2.5 s |
| Q-Learning optimisé | **13.0** | **+8.0** | **100 %** | ~2.3 s |
| SARSA optimisé | ~13.0 | ~+8.0 | ~100 % | ~10 s |
| Deep Q-Learning optimisé | ~13.7 | ~+7.2 | ~99.5 % | ~290 s |

Voir `RAPPORT.md` pour le détail méthodologique, la stratégie
d'optimisation des hyperparamètres et la discussion sur le reward shaping.

## Remarques

- `BruteForceAgent` désactive volontairement le `TimeLimit` Gymnasium par
  défaut (`max_episode_steps=3000`) pour mesurer le coût *réel* d'une
  politique aléatoire (~1800 steps) plutôt que la troncature à 200. Les
  algorithmes appris gardent la TimeLimit standard de 200 : sans impact
  puisqu'ils convergent à ~13 steps.
- Le DQN nécessite PyTorch ≥ 2.0 et Triton ≥ 3.0 (sur torch ≥ 2.6 certains
  appels backward importent `triton.backends.compiler`).
- L'historique est stocké dans `server/history.db` (SQLite). Pour repartir
  de zéro, supprimer ce fichier ou utiliser le bouton *Vider l'historique*.
