"""
TAXI DRIVER AGENT BRUTEFORCE

Objectif :
  Créer un agent qui résout Taxi-v3 avec des RÈGLES CODÉES À LA MAIN.
  Cet agent sera notre BASELINE, l'objectif est que Q-Learning fasse MIEUX.

Concept :
  Le bruteforce n'apprend rien. Il applique simplement des règles :
    1. Si pas de passager → aller prendre le passager
    2. Si passager à bord → aller déposer le passager
    3. Sinon → bouger aléatoirement

Complexité :
  - Calcul : O(1) par étape
  - Mémoire : O(1) — pas de Q-Table, pas de réseau
  - Apprentissage : Non (règles fixes)

"""

import gymnasium as gym
import numpy as np
import time
import json
import random
from pathlib import Path

# Générateur aléatoire dédié au bruteforce (seedable pour reproductibilité).
_rng = random.Random()


def set_seed(seed):
    """
    Fixe la graine du générateur aléatoire pour des benchmarks reproductibles.

    Args:
        seed (int | None): graine à utiliser, ou None pour de l'aléatoire pur.
    """
    if seed is not None:
        _rng.seed(seed)


def mouvement_aleatoire():
    """
    Tire un mouvement UNIFORME parmi les 4 directions [0-3].

    À ne pas confondre avec `env.action_space.sample() % 4` : ce dernier est
    biaisé (Sud=0 et Nord=1 sortent 2 fois plus souvent que Est=2 / Ouest=3,
    car sample() tire 0-5 et 4→0, 5→1 après le modulo).

    Returns:
        int: action de déplacement [0-3], chacune avec probabilité 1/4.
    """
    return _rng.randint(0, 3)


# SECTION 1 : CONFIGURATION DE L'ENVIRONNEMENT

"""
Taxi-v3 : grille 5×5, 4 lieux nommés (R, G, Y, B)

  État = (row, col, passenger, destination)
    row, col        : position du taxi [0-4]
    passenger       : lieu du passager à chercher [0-3] ou 4 si dans le taxi
    destination     : lieu de dépôt cible [0-3]

  Actions :
    0 = Déplacer Sud (down)
    1 = Déplacer Nord (up)
    2 = Déplacer Est  (right)
    3 = Déplacer Ouest (left)
    4 = Prendre le passager (pickup)
    5 = Déposer le passager (dropoff)

  Récompenses :
    +20 : mission complète (passager déposé au bon endroit)
    -1  : chaque étape (pour minimiser les steps)
    -10 : action illégale (ex: pickup sans passager sur la case)
"""

def initialiser_environnement(render_mode, max_steps=200):
    """
    Crée et initialise l'environnement Taxi-v3.

    Args:
        render_mode (str): "human" (fenêtre) ou "ansi" (terminal)
        max_steps (int): nombre max d'étapes avant troncature de l'épisode.
                         Par défaut Gymnasium plafonne Taxi-v3 à 200 ; on le
                         rend configurable pour reproduire le bruteforce naïf
                         à ~350 steps mentionné dans le sujet.

    Returns:
        gym.Env: environnement configuré
    """
    return gym.make("Taxi-v4", render_mode=render_mode, max_episode_steps=max_steps)


# SECTION 2 : DÉCODER L'ÉTAT

# Positions fixes des 4 lieux dans la grille 5×5
LIEUX = {
    0: (0, 0),  # Rouge (haut-gauche)
    1: (0, 4),  # Vert  (haut-droite)
    2: (4, 0),  # Jaune (bas-gauche)
    3: (4, 3),  # Bleu  (bas-droite)
}

NOMS_LIEUX = {
    0: "Rouge",
    1: "Vert",
    2: "Jaune",
    3: "Bleu",
}

ACTIONS = {
    0: "Sud",
    1: "Nord",
    2: "Est",
    3: "Ouest",
    4: "Pickup",
    5: "Dropoff",
}


def decoder_etat(state, env):
    """
    Déconde l'état entier (0-499) en coordonnées lisibles.

    Taxi-v3 encode l'état en un seul entier :
      state = row * 100 + col * 20 + passenger * 4 + destination

    D'où:
      row         : position Y du taxi [0-4]
      col         : position X du taxi [0-4]
      passenger   : 0-3 = lieu du passager, 4 = dans le taxi
      destination : 0-3 = lieu de dépôt cible

    Args:
        state (int): état codé [0-499]
        env: environnement Gymnasium

    Returns:
        tuple: (row, col, passenger, destination)
    """
    row, col, passenger, destination = env.unwrapped.decode(state)
    return row, col, passenger, destination


def afficher_etat(state, env):
    """
    Affiche l'état de manière lisible en français.

    Args:
        state (int): état codé
        env: environnement
    """
    row, col, passenger, destination = decoder_etat(state, env)

    if passenger < 4:
        pass_info = f"à récupérer au {NOMS_LIEUX[passenger]}"
    else:
        pass_info = f"à bord (destination: {NOMS_LIEUX[destination]})"

    print(f"    Position taxi: ({row},{col}) | Passager: {pass_info}")


# SECTION 3 : LOGIQUE DU BRUTEFORCE

"""
Règles d'or du Bruteforce :

  PHASE 1 : Chercher le passager
  ├─ Si on est sur la case du passager → PICKUP
  ├─ Sinon → mouvement aléatoire (0-3)

  PHASE 2 : Déposer le passager
  ├─ Si on est sur la destination → DROPOFF
  ├─ Sinon → mouvement aléatoire (0-3)

Aucune intelligence : pas de pathfinding, pas de stratégie.
C'est "essayer au hasard jusqu'à ce que ça marche".
"""


def bruteforce_action(state, env):
    """
    Décide une action selon les règles brutales du bruteforce.

    Logique :
      1. Décoder l'état
      2. Si passager pas en taxi (passenger < 4)
         → Si on est sur sa case : PICKUP
         → Sinon : mouvement aléatoire
      3. Si passager en taxi (passenger == 4)
         → Si on est sur la destination : DROPOFF
         → Sinon : mouvement aléatoire

    Args:
        state (int): état actuel
        env: environnement

    Returns:
        int: action à effectuer [0-5]
    """
    # Décoder l'état
    row, col, passenger, destination = decoder_etat(state, env)

    # PHASE 1 : Chercher le passager
    if passenger < 4:
        # Le passager est toujours au même lieu (fixe)
        lieu_passager_row, lieu_passager_col = LIEUX[passenger]

        # Si le taxi est sur la case du passager
        if row == lieu_passager_row and col == lieu_passager_col:
            return 4  # PICKUP
        else:
            # Sinon : bouger aléatoirement (0-3 = directions, pas pickup/dropoff)
            return mouvement_aleatoire()

    # PHASE 2 : Déposer le passager
    else:  # passenger == 4 (passager à bord)
        # Destination est la case cible
        dest_row, dest_col = LIEUX[destination]

        # Si le taxi est sur la destination
        if row == dest_row and col == dest_col:
            return 5  # DROPOFF
        else:
            # Sinon : bouger aléatoirement
            return mouvement_aleatoire()


# SECTION 4 : BOUCLE D'ENTRAÎNEMENT

"""
Structure d'un épisode :
  1. env.reset() → état initial, infos
  2. Boucle tant que not done :
     a. Choisir action
     b. env.step(action) → new_state, reward, terminated, truncated, info
     c. done = terminated or truncated
     d. Collecter métriques
  3. env.close()

Le bruteforce ne maintient aucun état d'apprentissage — c'est stateless.
"""


def lancer_episode_bruteforce(env, episode_num, afficher_etapes=False, sleep_time=0.0):
    """
    Lance un seul épisode du bruteforce.

    Args:
        env: environnement Gymnasium
        episode_num (int): numéro de l'épisode (pour affichage)
        afficher_etapes (bool): si True, affiche chaque étape
        sleep_time (float): délai entre chaque step (0.0 = pas de pause)

    Returns:
        dict: {'steps': int, 'reward': float, 'victoire': bool, 'temps': float}
    """
    state, info = env.reset()
    done = False
    terminated = False
    total_reward = 0
    nb_steps = 0

    if afficher_etapes:
        print(f"\n  ┌─ Épisode {episode_num} ─")

    # Chronomètre l'épisode (temps de résolution exigé par le sujet).
    # Mesure fiable uniquement quand sleep_time == 0 (benchmark sans pause).
    debut = time.perf_counter()

    while not done:
        # Choisir action selon les règles du bruteforce
        action = bruteforce_action(state, env)

        # Exécuter l'action dans l'environnement
        new_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Collecter les métriques
        total_reward += reward
        nb_steps += 1

        # Affichage détaillé (optionnel, ralentit l'exécution)
        if afficher_etapes and nb_steps <= 10:
            print(f"  │ Step {nb_steps:3d} | Action: {ACTIONS[action]:8s} | "
                  f"Reward: {reward:+3d} | Total: {total_reward:+5.0f}")

        # Pause uniquement si demandée (pour visualisation fenêtre)
        if sleep_time > 0:
            time.sleep(sleep_time)

        # Passer à l'état suivant
        state = new_state

    temps_episode = time.perf_counter() - debut

    # Victoire = l'épisode s'est terminé par un dépôt réussi (terminated=True),
    # et non par une troncature (truncated=True, limite de steps atteinte).
    # Plus robuste que de tester reward == 20 (valeur magique).
    victoire = terminated

    if afficher_etapes:
        status = "✓ VICTOIRE" if victoire else "✗ ÉCHEC"
        print(f"  └─ {status} | Steps: {nb_steps} | Reward total: {total_reward}\n")

    return {
        'steps': nb_steps,
        'reward': total_reward,
        'victoire': victoire,
        'temps': temps_episode,
    }


def lancer_bruteforce(nb_episodes, afficher_etapes=False, render_mode="human",
                      sleep_time=0.0, seed=None, max_steps=200):
    """
    Lance le bruteforce sur N épisodes et collecte les statistiques.

    Args:
        nb_episodes (int): nombre d'épisodes à jouer
        afficher_etapes (bool): afficher détails de chaque étape
        render_mode (str): "human" ou "ansi"
        sleep_time (float): délai entre chaque step pour visualisation
        seed (int | None): graine pour des résultats reproductibles
        max_steps (int): limite de steps par épisode (troncature)

    Returns:
        dict: statistiques complètes
    """
    set_seed(seed)
    env = initialiser_environnement(render_mode, max_steps=max_steps)

    # Seeder l'environnement une fois suffit : les états initiaux de chaque
    # env.reset() suivant découlent de façon déterministe de cette graine.
    # Sans ça, seuls les mouvements seraient reproductibles, pas les départs.
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)

    # Listes pour collecter les métriques
    steps_liste = []
    rewards_liste = []
    temps_liste = []
    victoires = 0

    print("═" * 65)
    print(f"BRUTEFORCE — {nb_episodes} épisodes")
    print("═" * 65)

    # Boucle d'entraînement
    for episode in range(nb_episodes):
        # Lancer un épisode
        resultat = lancer_episode_bruteforce(env, episode + 1,
                                            afficher_etapes=(afficher_etapes and episode < 3),
                                            sleep_time=sleep_time)

        steps_liste.append(resultat['steps'])
        rewards_liste.append(resultat['reward'])
        temps_liste.append(resultat['temps'])
        if resultat['victoire']:
            victoires += 1

        # Affichage du progrès toutes les 10 épisodes
        if (episode + 1) % 10 == 0 or episode == 0:
            win_rate = (victoires / (episode + 1)) * 100
            print(f"  Épisode {episode+1:4d} | Steps: {resultat['steps']:4d} | "
                  f"Reward: {resultat['reward']:+7.1f} | Win rate: {win_rate:.1f}%")

    env.close()

    # Convertir les listes en numpy arrays pour les calculs et le JSON
    steps_array = np.array(steps_liste)
    rewards_array = np.array(rewards_liste)
    temps_array = np.array(temps_liste)

    # Calculer les statistiques
    stats = {
        'nb_episodes': nb_episodes,
        'victoires': victoires,
        'win_rate': (victoires / nb_episodes) * 100,
        'steps_moyen': np.mean(steps_array),
        'steps_min': np.min(steps_array),
        'steps_max': np.max(steps_array),
        'steps_std': np.std(steps_array),
        'reward_moyen': np.mean(rewards_array),
        'reward_min': np.min(rewards_array),
        'reward_max': np.max(rewards_array),
        'reward_std': np.std(rewards_array),
        'temps_moyen': np.mean(temps_array),
        'temps_total': np.sum(temps_array),
        'temps_std': np.std(temps_array),
        'steps_liste': steps_array,
        'rewards_liste': rewards_array,
        'temps_liste': temps_array,
    }

    return stats


# SECTION 5 : AFFICHAGE DES RÉSULTATS

def afficher_resultats(stats):
    """
    Affiche les résultats de manière lisible et organisée.

    Args:
        stats (dict): statistiques retournées par lancer_bruteforce()
    """
    print()
    print("═" * 65)
    print("RÉSULTATS BRUTEFORCE — STATISTIQUES COMPLÈTES")
    print("═" * 65)

    print()
    print("RÉSUMÉ GÉNÉRAL")
    print("─" * 65)
    print(f"  Épisodes joués          : {stats['nb_episodes']}")
    print(f"  Missions réussies       : {stats['victoires']} / {stats['nb_episodes']}")
    print(f"  Win Rate (succès)       : {stats['win_rate']:.1f}%")

    print()
    print(" NOMBRE D'ÉTAPES (Steps)")
    print("─" * 65)
    print(f"  Moyenne                 : {stats['steps_moyen']:.1f} steps")
    print(f"  Minimum                 : {stats['steps_min']} steps")
    print(f"  Maximum                 : {stats['steps_max']} steps")
    print(f"  Écart-type              : {stats['steps_std']:.1f} steps")

    print()
    print(" RÉCOMPENSES (Rewards)")
    print("─" * 65)
    print(f"  Moyenne                 : {stats['reward_moyen']:.1f}")
    print(f"  Minimum                 : {stats['reward_min']:.1f}")
    print(f"  Maximum                 : {stats['reward_max']:.1f}")
    print(f"  Écart-type              : {stats['reward_std']:.1f}")

    print()
    print(" TEMPS DE RÉSOLUTION (exigé par le sujet)")
    print("─" * 65)
    print(f"  Temps moyen / épisode   : {stats['temps_moyen'] * 1000:.2f} ms")
    print(f"  Temps total             : {stats['temps_total']:.3f} s")
    print(f"  Écart-type              : {stats['temps_std'] * 1000:.2f} ms")

    print()
    print("INTERPRÉTATION")
    print("─" * 65)
    print(f"  ✓ C'est notre BASELINE — le point de référence")
    print(f"  ✓ Q-Learning devra faire MIEUX")
    print(f"  ✓ Win rate: {stats['win_rate']:.1f}% signifie {stats['victoires']} réussites sur {stats['nb_episodes']}")
    print(f"  ✓ Reward moyen: {stats['reward_moyen']:.1f} = chaque étape coûte -1")

    print()


def sauvegarder_resultats(stats, nom_fichier="bruteforce_stats.json"):
    """
    Sauvegarde les statistiques dans un fichier JSON pour les graphes.

    Args:
        stats (dict): statistiques
        nom_fichier (str): chemin du fichier JSON
    """
    # Convertir les arrays numpy en listes pour JSON
    # Aussi convertir les numpy types en types Python natifs pour JSON serialization
    stats_json = {}
    for key, value in stats.items():
        if isinstance(value, np.ndarray):
            # Les steps et rewards sont des entiers ; les temps restent des
            # flottants (sinon des durées < 1s seraient tronquées à 0).
            if key in ('steps_liste', 'rewards_liste'):
                stats_json[key] = value.astype(int).tolist()
            elif key.endswith('liste'):
                stats_json[key] = value.astype(float).tolist()
            else:
                stats_json[key] = float(value)
        elif isinstance(value, (np.integer, np.floating)):
            # Convertir numpy scalar → type Python
            stats_json[key] = int(value) if isinstance(value, np.integer) else float(value)
        else:
            # Garder les types Python tels quels
            stats_json[key] = value

    chemin = Path(__file__).parent / nom_fichier
    with open(chemin, 'w') as f:
        json.dump(stats_json, f, indent=2)

    print(f"✓ Résultats sauvegardés : {chemin}")


# SECTION 6 : INTERFACE UTILISATEUR (USER MODE vs TIME-LIMITED MODE)

def demander_mode():
    """
    Demande à l'utilisateur quel mode il veut.

    Returns:
        int: 1 (user mode) ou 2 (time-limited mode)
    """
    print()
    print("═" * 65)
    print("TAXI DRIVER — AGENT BRUTEFORCE")
    print("═" * 65)
    print()
    print("Choisissez un mode :")
    print("  1 → User mode       (vous choisissez les paramètres)")
    print("  2 → Time-limited mode (paramètres optimisés)")
    print()

    while True:
        try:
            choix = int(input("Votre choix (1 ou 2) : "))
            if choix in [1, 2]:
                return choix
            else:
                print("  ❌ Veuillez entrer 1 ou 2")
        except ValueError:
            print("  ❌ Veuillez entrer un nombre")


def mode_utilisateur():
    """
    Mode interactif : l'utilisateur choisit le nombre d'épisodes.

    Returns:
        dict: paramètres saisies par l'utilisateur
    """
    print()
    print("─" * 65)
    print("USER MODE — Configurez les paramètres")
    print("─" * 65)
    print()

    while True:
        try:
            nb_episodes = int(input("  Nombre d'épisodes à jouer (ex: 100) : "))
            if nb_episodes > 0:
                break
            else:
                print("  ❌ Le nombre doit être > 0")
        except ValueError:
            print("  ❌ Veuillez entrer un nombre")

    afficher_etapes = input("  Afficher les détails des 3 premiers épisodes ? (o/n) : ").lower() == 'o'
    demo = input("  Lancer la démo visuelle (fenêtre, 3 épisodes) ? (o/n) : ").lower() == 'o'

    # Seed optionnel pour des résultats reproductibles (Entrée = aléatoire).
    saisie_seed = input("  Graine aléatoire (seed) ? (nombre, ou Entrée pour aucun) : ").strip()
    seed = int(saisie_seed) if saisie_seed.isdigit() else None

    return {
        'nb_episodes': nb_episodes,
        'afficher_etapes': afficher_etapes,
        'sleep_time': 0.05,  # pause visible pour visualiser dans la fenêtre
        'seed': seed,
        'max_steps': 200,
        'demo': demo,
    }


def mode_temps_limite():
    """
    Mode temps-limité : paramètres optimisés par défaut.

    Pas de pause entre les steps — entraînement rapide (~5 secondes).
    La fenêtre graphique reste ouverte mais ne ralentit pas le code.

    Returns:
        dict: paramètres optimisés
    """
    print()
    print("─" * 65)
    print("TIME-LIMITED MODE — Paramètres optimisés")
    print("─" * 65)
    print()

    params = {
        'nb_episodes': 100,
        'afficher_etapes': False,
        'sleep_time': 0.0,  # pas de pause = exécution rapide
        'seed': 42,         # graine fixe = benchmark reproductible
        'max_steps': 200,
        'demo': False,      # pas de fenêtre = on respecte le budget temps
    }

    print(f"  Épisodes             : {params['nb_episodes']}")
    print(f"  Seed (reproductible) : {params['seed']}")
    print(f"  Temps estimé         : ~5 secondes")
    print()

    return params


# SECTION 7 : POINT D'ENTRÉE PRINCIPAL

def main():
    """
    Fonction principale — orchestre tout le programme.
    """
    # Demander le mode à l'utilisateur
    choix_mode = demander_mode()

    # Charger les paramètres en fonction du mode
    if choix_mode == 1:
        params = mode_utilisateur()
    else:
        params = mode_temps_limite()

    # Exécution sans fenêtre graphique (rapide, c'est ce qui est mesuré)
    print("  Exécution du bruteforce (sans fenêtre)...")
    stats = lancer_bruteforce(
        nb_episodes=params['nb_episodes'],
        afficher_etapes=params['afficher_etapes'],
        render_mode="ansi",
        sleep_time=0.0,
        seed=params['seed'],
        max_steps=params['max_steps'],
    )

    # Démonstration visuelle optionnelle : rejouer 3 épisodes avec la fenêtre.
    # Encadrée par try/except car le rendu "human" échoue en environnement
    # sans affichage (serveur, CI…) — on ne veut pas perdre les résultats.
    if params.get('demo'):
        print()
        print("  Démonstration visuelle — 3 épisodes (fenêtre graphique)...")
        try:
            lancer_bruteforce(
                nb_episodes=3,
                afficher_etapes=False,
                render_mode="human",
                sleep_time=0.3,
                max_steps=params['max_steps'],
            )
        except Exception as e:
            print(f"  ⚠️  Démo visuelle indisponible ({e}). On continue.")

    # Afficher les résultats
    afficher_resultats(stats)

    # Sauvegarder pour les graphes
    sauvegarder_resultats(stats)

    print()
    print("✓ Programme terminé. Résultats sauvegardés.")
    print("  → Utilisez 'python bruteforce_graphs.py' pour générer les graphes.")
    print()


if __name__ == "__main__":
    main()
