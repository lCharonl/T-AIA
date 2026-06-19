# PARTIE 2 — Agent Bruteforce sur Taxi-v3

---

## Objectif

Créer un agent qui résout Taxi-v3 en utilisant **des règles codées à la main**, sans apprentissage.

Le bruteforce sert de **BASELINE** — un point de référence pour mesurer si Q-Learning fait mieux.

---

## Qu'est-ce que le Bruteforce ?

```
Définition : appliquer des règles simples et codées manuellement,
            sans intelligence ni adaptation.

Exemple : "Si le passager n'est pas à bord ET je suis sur sa case,
          alors PICKUP. Sinon, bouge au hasard."
```

### Caractéristiques

| Propriété | Valeur |
|-----------|--------|
| **Apprentissage** | Non — règles fixes |
| **Mémoire** | O(1) — aucune Q-Table |
| **Intelligence** | 0 — très stupide |
| **Rapidité** | Très rapide (pas de calculs) |
| **Performance attendue** | Mauvaise (~26% win rate, cap 200 steps) |
| **Reproductibilité** | Oui via `seed` (env + mouvements seedés) |

---

## Les règles du Bruteforce

```
┌─────────────────────────────────────────────┐
│   ALGORITHME DU BRUTEFORCE                  │
└─────────────────────────────────────────────┘

  ÉTAPE 1 : Lire l'état
  └─ Position taxi : (row, col)
  └─ Passager : lieu à chercher [0-3] OU 4 si à bord
  └─ Destination : lieu de dépôt [0-3]

  ÉTAPE 2 : Décider l'action
  ├─ Si passager PAS en taxi (passenger < 4)
  │  ├─ Si je suis sur la case du passager
  │  │  └─ ACTION : PICKUP (4)
  │  └─ Sinon
  │     └─ ACTION : mouvement aléatoire (0-3)
  │
  └─ Si passager EN taxi (passenger == 4)
     ├─ Si je suis sur la destination
     │  └─ ACTION : DROPOFF (5)
     └─ Sinon
        └─ ACTION : mouvement aléatoire (0-3)

  ÉTAPE 3 : Boucler
  └─ Répéter jusqu'à victoire ou trop de steps
```

### Traduction en code

```python
def bruteforce_action(state, env):
    row, col, passenger, destination = decoder_etat(state, env)
    
    if passenger < 4:
        # Phase 1 : chercher le passager
        lieu_passager_row, lieu_passager_col = LIEUX[passenger]
        if row == lieu_passager_row and col == lieu_passager_col:
            return 4  # PICKUP
        else:
            return mouvement_aleatoire()  # mouvement UNIFORME 0-3
    else:
        # Phase 2 : déposer le passager
        dest_row, dest_col = LIEUX[destination]
        if row == dest_row and col == dest_col:
            return 5  # DROPOFF
        else:
            return mouvement_aleatoire()  # mouvement UNIFORME 0-3
```

> ⚠️ **Note technique** : on n'utilise PAS `env.action_space.sample() % 4`.
> `sample()` tire 0-5 uniformément, et le modulo renvoie 4→0 (Sud) et 5→1
> (Nord) : les déplacements Sud/Nord sortiraient alors **2× plus souvent** que
> Est/Ouest. `mouvement_aleatoire()` garantit une vraie probabilité 1/4 par
> direction — indispensable pour une baseline « hasard pur » honnête.

---

## Pourquoi c'est faible ?

Le bruteforce ne peut PAS optimiser sa route.

```
Situation réelle :
  Taxi à (0,0), passager à (4,3), destination (0,4)

Bruteforce :
  "Je suis pas sur la case du passager... je bouge au hasard!"
  → Peut aller n'importe où (haut, bas, gauche, droite, même reculer)
  → Entièrement dépendant de la chance

Q-Learning :
  "J'ai appris que pour aller vers (4,3), je dois aller SUD"
  → Optimal, raisonnée, efficace
```

**Résultat** : le bruteforce prend ~180 steps tandis que Q-Learning en prend ~13.

---

## Structure du code

Le code est organisé en **7 sections** :

### Section 1 : Configuration de l'environnement
```python
def initialiser_environnement(render_mode):
    return gym.make("Taxi-v3", render_mode=render_mode)
```

### Section 2 : Décoder l'état
```python
# État codé en entier : state = row * 100 + col * 20 + passenger * 4 + destination
# On récupère : (row, col, passenger, destination)

LIEUX = {
    0: (0, 0),  # Rouge
    1: (0, 4),  # Vert
    2: (4, 0),  # Jaune
    3: (4, 3),  # Bleu
}
```

### Section 3 : Logique du bruteforce
```python
def bruteforce_action(state, env):
    # Règles codées à la main
    # Pas d'apprentissage, pas de Q-Table
```

### Section 4 : Boucle d'entraînement
```python
def lancer_episode_bruteforce(env, episode_num):
    # 1 épisode = 1 tentative complète
    # Collecter steps et rewards
```

### Section 5 : Affichage des résultats
```python
def afficher_resultats(stats):
    # Tableau avec toutes les métriques
```

### Section 6 : Modes (User vs Time-Limited)
```
User mode : l'utilisateur choisit le nombre d'épisodes
Time-limited mode : 100 épisodes d'office (rapide)
```

### Section 7 : Point d'entrée principal
```python
def main():
    # Orchestrer tout
```

---

## Les deux modes

### Mode 1 : User Mode 

L'utilisateur contrôle les paramètres :

```
Votre choix (1 ou 2) : 1

USER MODE — Configurez les paramètres

  Nombre d'épisodes à jouer (ex: 100) : 200
  Afficher les détails des 3 premiers épisodes ? (o/n) : o
  Lancer la démo visuelle (fenêtre, 3 épisodes) ? (o/n) : n
  Graine aléatoire (seed) ? (nombre, ou Entrée pour aucun) : 42
```

**Intérêt** : tester avec différents nombres d'épisodes, afficher les détails.

### Mode 2 : Time-Limited Mode

Paramètres optimisés par défaut :

```
Votre choix (1 ou 2) : 2

TIME-LIMITED MODE — Paramètres optimisés

  Épisodes             : 100
  Seed (reproductible) : 42
  Temps estimé         : ~5 secondes
```

**Intérêt** : résultats rapides et fiables.

---

## Résultats typiques

Après 100 épisodes du bruteforce (**seed=42, max_steps=200** → reproductible) :

### Statistiques clés

```
Épisodes joués          : 100
Missions réussies       : 26 / 100
Win Rate (succès)       : 26%

Nombre d'étapes (Steps)
  Moyenne               : 179.2 steps
  Minimum               : 30 steps
  Maximum               : 200 steps  (= limite max_steps)
  Écart-type            : 42.2 steps

Récompenses (Rewards)
  Moyenne               : -173.8
  Minimum               : -200.0
  Maximum               : -9.0
  Écart-type            : 50.1

Temps de résolution (exigé par le sujet)
  Temps moyen / épisode : ~2.3 ms
  Temps total (100 ep.) : ~0.23 s
```

> ℹ️ **Le maximum est plafonné à 200** car `gym.make("Taxi-v3")` applique par
> défaut un `TimeLimit` à 200 steps (l'épisode est tronqué, pas gagné). Pour
> reproduire le bruteforce naïf à **~350 steps** évoqué dans le sujet, on lance
> avec `max_steps=500` : le taux de réussite remonte (~80%) et la moyenne grimpe
> vers ~270 steps, car le taxi a le droit d'errer plus longtemps avant de réussir.

### Interprétation

```
✓ 26% win rate = très faible (26 victoires sur 100)

✓ 179 steps en moyenne = beaucoup trop
  (optimal = 13 steps dans la partie 3)

✓ Reward -173.8 = très négatif
  Chaque step coûte -1, donc 179 steps ≈ -179
  Plus la pénalité -10 pour actions illégales

✓ Max = 200 steps = la limite (TimeLimit)
  = le taxi s'est perdu jusqu'à la troncature !

✓ Temps ~2.3 ms/épisode = ultra rapide
  (aucun calcul lourd : juste des règles O(1))

✓ Écart-type élevé = très variable
  Certaines parties réussissent en 30 steps, d'autres
  errent jusqu'à 200. Aucune cohérence.
```

---

## Graphiques générés

Le script `bruteforce_graphs.py` génère **7 graphiques** :

### Graphique 1 : Évolution des steps par épisode
```
Courbe montrant comment le nombre de steps varie
d'un épisode à l'autre.

Attendu : aucun pattern — complètement aléatoire
          car le bruteforce n'apprend pas
```

### Graphique 2 : Évolution du reward par épisode
```
Courbe montrant comment le reward varie.

Attendu : mostly négatif, quelques pics positifs (+20)
          quand une mission réussit par chance
```

### Graphique 3 : Distribution des steps (histogramme)
```
Combien d'épisodes ont pris 20 steps ?
Combien en ont pris 100 ?
Combien en ont pris 400 ?

Attendu : très étalé, aucune concentration
```

### Graphique 4 : Distribution des rewards (histogramme)
```
Combien d'épisodes ont un reward de -150 ?
Combien ont un reward de -200 ?
Combien ont un reward de +20 ?

Attendu : pic autour de -170, quelques rares +20
```

### Graphique 5 : Résumé texte
```
Toutes les métriques clés en un seul graphe textuel
```

### Graphique 6 : Boîtes à moustaches
```
Visualise la distribution statistique.

Boîte = zone centrale (50% des valeurs)
Moustaches = min et max
```

### Graphique 7 : Temps de résolution (`bruteforce_temps.png`)
```
À gauche  : temps (ms) par épisode + moyenne
À droite  : distribution (histogramme) des temps

Métrique EXIGÉE par le sujet ("mean time for finishing
the game"). Montre que le bruteforce est trivialement
rapide en calcul (~2 ms/épisode) — sa faiblesse est le
NOMBRE de steps, pas le temps machine.
```

---

## Comment utiliser le code

### Étape 1 : Lancer le bruteforce

```bash
python taxi_step2_bruteforce.py
```

Choisir le mode :
- Mode 1 : configurez vous-même
- Mode 2 : rapide (recommandé)

### Étape 2 : Afficher les graphes

```bash
python bruteforce_graphs.py
```

Génère 4 fichiers PNG :
- `bruteforce_graphs.png` — 4 graphiques (steps & rewards)
- `bruteforce_resume.png` — résumé texte
- `bruteforce_boites.png` — boîtes à moustaches
- `bruteforce_temps.png` — temps de résolution par épisode

### Étape 3 : Analyser

Regarder les graphes et se poser des questions :

```
Q : Pourquoi la moyenne est 170 et pas 13 ?
R : Car le bruteforce bouge au hasard, pas optimal

Q : Pourquoi seulement 30% de succès ?
R : Être au hasard, on peut tomber dans un piège
    ou prendre trop de steps (limite 500)

Q : Est-ce que le bruteforce apprend ?
R : Non, aucune courbe d'apprentissage — tout aléatoire

Q : Comment Q-Learning ferait mieux ?
R : En apprenant les meilleures actions et les rejouer
```

---

## Complexité

### Complexité temporelle

```
Par épisode :
  O(steps_moyen) = O(170)

N épisodes :
  O(N × 170)

100 épisodes :
  O(17 000) étapes total
```

Très rapide car chaque étape est O(1) — juste une règle.

### Complexité spatiale

```
O(1) — aucune structure de données

Le bruteforce ne garde en mémoire que :
  - État courant (entier)
  - Reward accumulé (float)

Pas de Q-Table, pas de réseau neural.
```

---

## Améliorations possibles

> ✅ **Déjà appliquées dans cette version :**
> - Mouvement aléatoire **uniforme** (correction du biais `% 4`)
> - Mesure du **temps de résolution** (métrique exigée par le sujet)
> - Détection de victoire via le flag `terminated` (au lieu de `reward == 20`)
> - **Reproductibilité** via `seed` (environnement + mouvements)
> - `max_steps` **configurable** (200 par défaut, 500 pour le benchmark naïf ~350)

Le bruteforce reste une **baseline**, pas une solution réelle. Pistes restantes :

### Amélioration 1 : Pathfinding

Ajouter A* pour trouver la route optimale.

```
Actuel : "Je bouge au hasard jusqu'à tomber sur le passager"
Amélioré : "Je calcule la route optimale et la suis"

Résultat : ~15-20 steps au lieu de 170
```

### Amélioration 2 : Heuristiques

Coder des règles "intelligentes" :

```
Exemple : "Je ne dois jamais aller vers les bordures"
Exemple : "Si j'ai déjà visité cette case, ne pas y retourner"

Résultat : mieux que 30% win rate
```

### Amélioration 3 : Apprentissage (Q-Learning)

C'est exactement ce qu'on fait dans la Partie 3 !

```
Au lieu de coder les règles → Les APPRENDRE
```

---

## Conclusion

Le bruteforce est intentionnellement mauvais.

**Objectif atteint** :

```
✓ Créé une baseline pour comparaison
✓ Montré qu'on peut résoudre Taxi-v3 sans apprentissage
✓ Mesuré la performance (30% win rate, 170 steps)
✓ Généré des graphes pour visualisation
✓ Documenté chaque partie du code
```

**Ensuite** : le Q-Learning va faire BEAUCOUP mieux.

---

## Fichiers associés

- `taxi_step2_bruteforce.py` — implémentation complète du bruteforce
- `bruteforce_graphs.py` — script pour générer les graphes
- `bruteforce_stats.json` — statistiques sauvegardées (créé après la première exécution)
- `bruteforce_graphs.png` — 4 graphiques de performance
- `bruteforce_resume.png` — résumé texte des résultats
- `bruteforce_boites.png` — boîtes à moustaches
- `bruteforce_temps.png` — temps de résolution par épisode
