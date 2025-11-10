import shap
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.text import Text
try:
    from shap.plots import style_context  # type: ignore # SHAP >= 0.45
except ImportError:  # pragma: no cover - rétrocompat
    from shap.plots._style import style_context
from matplotlib.colors import Colormap, LinearSegmentedColormap

from projet_05.branding import Theme, apply_brand_theme

apply_brand_theme()
DEFAULT_CMAP = Theme.colormap(
    "diverging",
    start="primary",
    end="secondary",
    blend_center=True,
    strong_ends=True,
)


warnings.filterwarnings("ignore", category=UserWarning)

def _resolve_cmap(user_cmap):
    """Retourne un objet Colormap utilisable par Matplotlib/SHAP."""
    if user_cmap is None:
        return DEFAULT_CMAP
    if isinstance(user_cmap, Colormap):
        return user_cmap
    if isinstance(user_cmap, str):
        try:
            return plt.get_cmap(user_cmap)
        except ValueError:
            print(f"[AVERTISSEMENT] Colormap '{user_cmap}' inconnue → utilisation du thème par défaut.")
            return DEFAULT_CMAP
    # Essaye de convertir une liste de couleurs
    if isinstance(user_cmap, (list, tuple)):
        try:
            return LinearSegmentedColormap.from_list("user_cmap", user_cmap)
        except Exception:
            print("[AVERTISSEMENT] Impossible de convertir la liste de couleurs fournie → colormap par défaut.")
            return DEFAULT_CMAP
    print("[AVERTISSEMENT] Objet colormap non reconnu → utilisation du thème par défaut.")
    return DEFAULT_CMAP


def shap_global(
    final_pipe,
    X_full,
    y_full,
    RANDOM_STATE=42,
    sample_size=400,
    plot_type="dot",
    max_display=15,
    cmap=None,
):
    """
    Affiche un diagramme SHAP global avec fallback automatique si le graphique principal échoue.
    """

    print("\n=== Interprétation SHAP globale ===")

    # --- Étape 1 : Entraînement global ---
    final_pipe.fit(X_full, y_full)
    fitted_model = final_pipe.named_steps['clf']

    # --- Étape 2 : Récupération noms de variables ---
    try:
        feature_names = final_pipe.named_steps['prep'].get_feature_names_out()
    except Exception:
        feature_names = list(X_full.columns)
    print(f"[INFO] {len(feature_names)} variables utilisées.")

    # --- Étape 3 : Préparation jeu d’échantillons ---
    X_sample = X_full.sample(n=min(sample_size, len(X_full)), random_state=RANDOM_STATE)
    X_transformed = final_pipe.named_steps['prep'].transform(X_sample)

    # Convertit toute structure hybride (sparse, object) vers un tableau float64
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()
    X_transformed = np.asarray(X_transformed)

    if not np.issubdtype(X_transformed.dtype, np.number):
        X_numeric = (
            pd.DataFrame(X_transformed)
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
        )
    else:
        X_numeric = X_transformed.astype(np.float64, copy=False)

    if X_numeric.ndim == 1:
        X_numeric = X_numeric.reshape(-1, 1)

    # --- Étape 4 : Vérifications ---
    print(f"[INFO] X_transformed shape: {X_numeric.shape}")
    if X_numeric.shape[1] == 0:
        print("[ERREUR] Aucune variable numérique n’a été conservée après transformation.")
        return None, X_numeric, feature_names

    if len(feature_names) != X_numeric.shape[1]:
        print("[AVERTISSEMENT] Noms de variables réindexés pour cohérence.")
        feature_names = [f"var_{i}" for i in range(X_numeric.shape[1])]

    X_df = pd.DataFrame(X_numeric, columns=feature_names).astype(np.float64, copy=False)

    # --- Étape 5 : Création de l’explainer ---
    try:
        explainer = shap.Explainer(fitted_model, X_df, feature_names=feature_names)
    except Exception:
        explainer = shap.Explainer(fitted_model.predict, X_df, feature_names=feature_names)

    shap_values = explainer(X_df)

    # Pour les modèles de classification multi-sorties, on conserve la classe positive
    if hasattr(shap_values, "values") and shap_values.values.ndim == 3:
        target_idx = 1 if shap_values.values.shape[-1] > 1 else 0
        shap_values = shap_values[..., target_idx]

    # --- Étape 6 : Nettoyage des valeurs ---
    if hasattr(shap_values, "values"):
        shap_array = np.nan_to_num(np.array(shap_values.values))
    else:
        shap_array = np.nan_to_num(np.array(shap_values))

    if shap_array.ndim == 1:
        shap_array = shap_array.reshape(-1, 1)

    print(f"[INFO] SHAP array shape: {shap_array.shape}")

    # --- Étape 7 : Diagramme ---
    summary_cmap = _resolve_cmap(cmap)
    try:
        shap.summary_plot(
            shap_values,
            features=X_df,
            feature_names=feature_names,
            plot_type=plot_type,
            max_display=max_display,
            show=False,
            cmap=summary_cmap,
            color=Theme.PRIMARY if plot_type == "bar" else None
        )
        fig = plt.gcf()
        ax = plt.gca()
        fig.patch.set_facecolor(Theme.BACKGROUND)
        ax.set_facecolor(Theme.BACKGROUND)
        plt.title(f"SHAP Summary Plot ({plot_type}) – Importance globale", fontsize=11)
        plt.tight_layout()
        plt.savefig(f"output/shap_summary_{plot_type}.png", dpi=300)
        plt.show()
        plt.close()
    except Exception as e:
        print(f"[AVERTISSEMENT] Plot détaillé échoué ({e}) → fallback barplot.")
        try:
            shap.summary_plot(
                shap_array.astype(np.float64, copy=False),
                features=X_df,
                feature_names=feature_names,
                plot_type="bar",
                max_display=max_display,
                show=False,
                color=Theme.PRIMARY
            )
            fig = plt.gcf()
            ax = plt.gca()
            fig.patch.set_facecolor(Theme.BACKGROUND)
            ax.set_facecolor(Theme.BACKGROUND)
            plt.title("SHAP Summary Plot – (fallback barplot)", fontsize=11)
            plt.tight_layout()
            plt.savefig(f"output/shap_summary_fallback_barplot.png", dpi=300)
            plt.show()
            plt.close()
        except Exception as e2:
            print(f"[ERREUR] Même le fallback échoue : {e2}")

    return shap_values, X_numeric, feature_names


# === Fonction 2 : SHAP LOCAL ===
def shap_local(idx, shap_values, max_display=10, text_scale=0.6, cmap=DEFAULT_CMAP):
    """
    Affiche un diagramme SHAP local plus lisible (textes réduits).
    
    Paramètres
    ----------
    idx : int
        Index de l'individu à expliquer.
    shap_values : shap.Explanation
        Valeurs SHAP calculées.
    max_display : int, par défaut 10
        Nombre max de variables à afficher.
    text_scale : float, par défaut 0.6
        Facteur d’échelle pour réduire la taille du texte.
    """

    idx = int(idx)
    n_obs = shap_values.shape[0]
    if idx < 0 or idx >= n_obs:
        print(f"[ERREUR] Index {idx} hors limites (0 ≤ idx < {n_obs}).")
        return

    print(f"\n[INFO] Interprétation locale SHAP – individu {idx}")

    # --- Crée la figure ---
    plt.figure(figsize=(7, 5))
    style_kwargs = dict(
        primary_color_positive=Theme.SECONDARY,
        primary_color_negative=Theme.PRIMARY,
        secondary_color_positive=Theme.SECONDARY,
        secondary_color_negative=Theme.PRIMARY,
        hlines_color=Theme.SECONDARY,
        vlines_color=Theme.SECONDARY,
        text_color="white",
        tick_labels_color="black",
    )

    # --- Génère le graphique SHAP ---
    with style_context(**style_kwargs):
        shap.plots.waterfall(shap_values[idx], max_display=max_display, show=False)
    fig = plt.gcf()
    ax = plt.gca()
    fig.patch.set_facecolor(Theme.BACKGROUND)
    ax.set_facecolor(Theme.BACKGROUND)

    # --- Ajuste la taille du texte ---
    for txt in plt.gcf().findobj(Text):
        size = txt.get_fontsize()
        try:
            size_f = float(size)
        except (TypeError, ValueError):
            # ignore les objets dont la taille n'est pas convertible
            continue
        txt.set_fontsize(size_f * text_scale)
        
    plt.title(f"Explication locale SHAP – individu {idx}", fontsize=9, pad=10)
    plt.tight_layout()
    plt.savefig(f"output/shap_local_individu_{idx}.png", dpi=300)
    plt.show()
    plt.close()
