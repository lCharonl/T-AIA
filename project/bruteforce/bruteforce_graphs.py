"""
BRUTEFORCE: GÉNÉRATION DES GRAPHIQUES

Ce script lit les statistiques sauvegardées par taxi_step2_bruteforce.py
et génère des graphiques visualisant la performance du bruteforce.

Graphiques générés:
  1. Évolution des steps par épisode (courbe)
  2. Évolution du reward par épisode (courbe)
  3. Distribution des steps (histogramme)
  4. Distribution des rewards (histogramme)
  5. Résumé avec annotations (boîte de texte)

"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Configuration matplotlib pour les graphes plus lisibles
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10
plt.rcParams['lines.linewidth'] = 2


def charger_stats(nom_fichier="bruteforce_stats.json"):
    """
    Charge les statistiques depuis le fichier JSON.

    Args:
        nom_fichier (str): nom du fichier JSON

    Returns:
        dict: statistiques chargées
    """
    chemin = Path(__file__).parent / nom_fichier

    if not chemin.exists():
        print(f"❌ Fichier {nom_fichier} non trouvé")
        print("   Lancez d'abord: python taxi_step2_bruteforce.py")
        return None

    with open(chemin, 'r') as f:
        stats = json.load(f)

    print(f"✓ Fichier chargé : {nom_fichier}")
    return stats


def creer_graphes(stats):
    """
    Crée 5 graphiques pour visualiser la performance du bruteforce.

    Args:
        stats (dict): statistiques chargées
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"BRUTEFORCE — Analyse des {stats['nb_episodes']} épisodes",
        fontsize=16, fontweight='bold', y=0.995
    )

    # GRAPHIQUE 1 : Évolution des steps par épisode
    ax1 = axes[0, 0]

    episodes = np.arange(1, stats['nb_episodes'] + 1)
    steps = np.array(stats['steps_liste'])

    ax1.plot(episodes, steps, linewidth=1.5, alpha=0.7, color='#1f77b4', label='Steps par épisode')
    ax1.axhline(y=stats['steps_moyen'], color='red', linestyle='--', linewidth=2, label=f"Moyenne: {stats['steps_moyen']:.1f}")
    ax1.fill_between(episodes, steps, alpha=0.2, color='#1f77b4')

    ax1.set_xlabel('Épisode', fontweight='bold')
    ax1.set_ylabel('Nombre de steps', fontweight='bold')
    ax1.set_title('Progression des steps par épisode', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # GRAPHIQUE 2 : Évolution du reward par épisode

    ax2 = axes[0, 1]

    rewards = np.array(stats['rewards_liste'])

    ax2.plot(episodes, rewards, linewidth=1.5, alpha=0.7, color='#2ca02c', label='Reward par épisode')
    ax2.axhline(y=stats['reward_moyen'], color='red', linestyle='--', linewidth=2, label=f"Moyenne: {stats['reward_moyen']:.1f}")
    ax2.fill_between(episodes, rewards, alpha=0.2, color='#2ca02c')

    ax2.set_xlabel('Épisode', fontweight='bold')
    ax2.set_ylabel('Reward total', fontweight='bold')
    ax2.set_title('Progression du reward par épisode', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # ─────────────────────────────────────────────────────────────────
    # GRAPHIQUE 3 : Distribution des steps (histogramme)
    # ─────────────────────────────────────────────────────────────────
    ax3 = axes[1, 0]

    ax3.hist(steps, bins=20, color='#1f77b4', alpha=0.7, edgecolor='black')
    ax3.axvline(x=stats['steps_moyen'], color='red', linestyle='--', linewidth=2, label=f"Moyenne: {stats['steps_moyen']:.1f}")
    ax3.axvline(x=stats['steps_min'], color='green', linestyle=':', linewidth=2, label=f"Min: {stats['steps_min']}")
    ax3.axvline(x=stats['steps_max'], color='orange', linestyle=':', linewidth=2, label=f"Max: {stats['steps_max']}")

    ax3.set_xlabel('Nombre de steps', fontweight='bold')
    ax3.set_ylabel('Fréquence', fontweight='bold')
    ax3.set_title('Distribution des steps', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.legend()

    # ─────────────────────────────────────────────────────────────────
    # GRAPHIQUE 4 : Distribution des rewards (histogramme)
    # ─────────────────────────────────────────────────────────────────
    ax4 = axes[1, 1]

    ax4.hist(rewards, bins=20, color='#2ca02c', alpha=0.7, edgecolor='black')
    ax4.axvline(x=stats['reward_moyen'], color='red', linestyle='--', linewidth=2, label=f"Moyenne: {stats['reward_moyen']:.1f}")
    ax4.axvline(x=stats['reward_min'], color='green', linestyle=':', linewidth=2, label=f"Min: {stats['reward_min']:.0f}")
    ax4.axvline(x=stats['reward_max'], color='orange', linestyle=':', linewidth=2, label=f"Max: {stats['reward_max']:.0f}")

    ax4.set_xlabel('Reward total', fontweight='bold')
    ax4.set_ylabel('Fréquence', fontweight='bold')
    ax4.set_title('Distribution des rewards', fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.legend()

    plt.tight_layout()

    # Sauvegarder la figure
    nom_sortie = Path(__file__).parent / "bruteforce_graphs.png"
    plt.savefig(nom_sortie, dpi=150, bbox_inches='tight')
    print(f"✓ Graphique sauvegardé : {nom_sortie}")

    plt.show()


def creer_graphe_resume(stats):
    """
    Crée un graphe de résumé avec les statistiques clés en texte.

    Args:
        stats (dict): statistiques chargées
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')

    # Titre
    fig.suptitle(
        "BRUTEFORCE — Résumé des Résultats",
        fontsize=18, fontweight='bold', y=0.98
    )

    # Bloc temps (optionnel : seulement si le JSON contient les nouvelles clés)
    if 'temps_moyen' in stats:
        bloc_temps = f"""
═══════════════════════════════════════════════════════════════

  TEMPS DE RÉSOLUTION

  Temps moyen / épisode       {stats['temps_moyen'] * 1000:.2f} ms
  Temps total                 {stats['temps_total']:.3f} s
  Écart-type                  {stats['temps_std'] * 1000:.2f} ms
"""
    else:
        bloc_temps = ""

    # Texte de résumé
    texte_resume = f"""
═══════════════════════════════════════════════════════════════

 RÉSULTATS BRUTEFORCE

  Épisodes joués              {stats['nb_episodes']}
  Missions réussies           {stats['victoires']} / {stats['nb_episodes']}
  Win Rate                    {stats['win_rate']:.1f}%

═══════════════════════════════════════════════════════════════

 NOMBRE D'ÉTAPES (Steps)

  Moyenne                     {stats['steps_moyen']:.1f} steps
  Minimum                     {stats['steps_min']} steps
  Maximum                     {stats['steps_max']} steps
  Écart-type                  {stats['steps_std']:.1f} steps

═══════════════════════════════════════════════════════════════

 RÉCOMPENSES (Rewards)

  Moyenne                     {stats['reward_moyen']:.1f}
  Minimum                     {stats['reward_min']:.1f}
  Maximum                     {stats['reward_max']:.1f}
  Écart-type                  {stats['reward_std']:.1f}
{bloc_temps}
═══════════════════════════════════════════════════════════════

 INTERPRÉTATION

  ✓ C'est notre BASELINE — le point de référence

  ✓ Q-Learning devra faire MIEUX

  ✓ Win rate: {stats['win_rate']:.1f}% = {stats['victoires']} réussites seulement

  ✓ Reward moyen: {stats['reward_moyen']:.1f} (négatif = pas optimisé)

═══════════════════════════════════════════════════════════════
    """

    ax.text(0.5, 0.5, texte_resume, transform=ax.transAxes,
            fontfamily='monospace', fontsize=11,
            verticalalignment='center', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Sauvegarder
    nom_sortie = Path(__file__).parent / "bruteforce_resume.png"
    plt.savefig(nom_sortie, dpi=150, bbox_inches='tight')
    print(f"✓ Résumé sauvegardé : {nom_sortie}")

    plt.show()


def creer_graphe_comparaison_boite(stats):
    """
    Crée un graphe "boîte à moustaches" pour visualiser la distribution.

    Args:
        stats (dict): statistiques chargées
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    fig.suptitle("BRUTEFORCE — Boîtes à moustaches (Distribution)",
                 fontsize=14, fontweight='bold')

    # Boîte steps
    bp1 = ax1.boxplot(stats['steps_liste'], vert=True, patch_artist=True)
    for patch in bp1['boxes']:
        patch.set_facecolor('lightblue')
    ax1.set_ylabel('Steps', fontweight='bold')
    ax1.set_title('Distribution des steps', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # Boîte rewards
    bp2 = ax2.boxplot(stats['rewards_liste'], vert=True, patch_artist=True)
    for patch in bp2['boxes']:
        patch.set_facecolor('lightgreen')
    ax2.set_ylabel('Reward', fontweight='bold')
    ax2.set_title('Distribution des rewards', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    nom_sortie = Path(__file__).parent / "bruteforce_boites.png"
    plt.savefig(nom_sortie, dpi=150, bbox_inches='tight')
    print(f"✓ Boîtes à moustaches sauvegardées : {nom_sortie}")

    plt.show()


def creer_graphe_temps(stats):
    """
    Crée 2 graphiques sur le TEMPS de résolution (métrique exigée par le sujet).

    Args:
        stats (dict): statistiques chargées
    """
    if 'temps_liste' not in stats:
        print("⚠️  Pas de données de temps dans le JSON (relancez le bruteforce).")
        return

    # Temps en millisecondes pour une lecture plus parlante
    temps_ms = np.array(stats['temps_liste']) * 1000
    episodes = np.arange(1, stats['nb_episodes'] + 1)
    temps_moyen_ms = stats['temps_moyen'] * 1000

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("BRUTEFORCE — Temps de résolution par épisode",
                 fontsize=14, fontweight='bold')

    # Courbe : temps par épisode
    ax1.plot(episodes, temps_ms, linewidth=1.5, alpha=0.7, color='#9467bd',
             label='Temps par épisode')
    ax1.axhline(y=temps_moyen_ms, color='red', linestyle='--', linewidth=2,
                label=f"Moyenne: {temps_moyen_ms:.2f} ms")
    ax1.fill_between(episodes, temps_ms, alpha=0.2, color='#9467bd')
    ax1.set_xlabel('Épisode', fontweight='bold')
    ax1.set_ylabel('Temps (ms)', fontweight='bold')
    ax1.set_title('Progression du temps par épisode', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Histogramme : distribution des temps
    ax2.hist(temps_ms, bins=20, color='#9467bd', alpha=0.7, edgecolor='black')
    ax2.axvline(x=temps_moyen_ms, color='red', linestyle='--', linewidth=2,
                label=f"Moyenne: {temps_moyen_ms:.2f} ms")
    ax2.set_xlabel('Temps (ms)', fontweight='bold')
    ax2.set_ylabel('Fréquence', fontweight='bold')
    ax2.set_title('Distribution des temps', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend()

    plt.tight_layout()

    nom_sortie = Path(__file__).parent / "bruteforce_temps.png"
    plt.savefig(nom_sortie, dpi=150, bbox_inches='tight')
    print(f"✓ Graphe du temps sauvegardé : {nom_sortie}")

    plt.show()


def main():
    """
    Fonction principale — génère tous les graphes.
    """
    print()
    print("═" * 70)
    print("BRUTEFORCE — GÉNÉRATION DES GRAPHIQUES")
    print("═" * 70)
    print()

    # Charger les stats
    stats = charger_stats()
    if stats is None:
        return

    print()
    print("Génération des graphiques...")
    print()

    # Créer les graphes
    creer_graphes(stats)
    creer_graphe_resume(stats)
    creer_graphe_comparaison_boite(stats)
    creer_graphe_temps(stats)

    print()
    print("✓ Tous les graphiques ont été générés et sauvegardés !")
    print()


if __name__ == "__main__":
    main()
