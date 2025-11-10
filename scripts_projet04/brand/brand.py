"""Palette et thèmes graphiques pour Matplotlib/Seaborn.

Ce module fournit une classe utilitaire (`Theme`) et une configuration
externe (`ThemeConfig`, `configure_theme`) permettant de définir des
couleurs, des palettes qualitatives et des cartes de couleurs (colormaps)
cohérentes. Des fonctions de démonstration et des wrappers
rétrocompatibles sont également fournis.
"""

from pathlib import Path
from typing import Any, List, Literal, Mapping, Optional, Tuple, Union
from dataclasses import dataclass, field, fields

import seaborn as sns
import numpy as np  # Données factices pour les démos
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


#
# Dataclass de configuration et gestion externe du thème
#
@dataclass
class ThemeConfig:
    """Configuration externe du thème.

    Cette structure regroupe les couleurs principales, les variantes de
    palette et des options d'apparence (fond, rcParams). Elle peut être
    passée à :func:`configure_theme`.

    Attributs
    ---------
    primary, secondary, tertiary : str
        Couleurs principales au format hexadécimal (p. ex. "#RRGGBB").
    background : str
        Couleur d'arrière‑plan des figures et axes.
    primary_variants, secondary_variants, tertiary_variants : list[str]
        Variantes qualitatives pour les séries multiples.
    sequential_light, sequential_dark : dict | None
        Remplacements explicites des teintes claires/foncées pour les
        colormaps séquentiels.
    light_amount, dark_amount : float
        Coefficients utilisés pour éclaircir/assombrir si aucun
        remplacement n'est fourni.
    text_color, axes_labelcolor, tick_color : str
        Couleurs des textes, étiquettes et graduations.
    figure_dpi, savefig_dpi : int
        Résolution d'affichage et d'export.
    """
    # Couleurs principales
    primary: str = "#7451EB"
    secondary: str = "#EE8273"
    tertiary: str = "#A6BD63"
    # Couleur d'arrière-plan
    background: str = "#FFFCF2"
    # Variantes qualitatives
    primary_variants: List[str] = field(default_factory=lambda: ["#9D7EF0", "#4B25D6"])
    secondary_variants: List[str] = field(default_factory=lambda: ["#F3A093", "#D95848"])
    tertiary_variants: List[str] = field(default_factory=lambda: ["#BDD681", "#7E923F"])
    # Remplacements stops séquentiels
    sequential_light: Optional[dict] = field(default_factory=lambda: {
        "primary": "#f3f0fd",
        "secondary": "#fdecea",
        "tertiary": "#f6faec",
    })
    sequential_dark: Optional[dict] = field(default_factory=lambda: {
        "primary": "#2f1577",
        "secondary": "#8b3025",
        "tertiary": "#4b5c27",
    })
    # Coefficients de mélange par défaut
    light_amount: float = 0.85
    dark_amount: float = 0.65
    # RC params (matplotlib)
    text_color: str = "black"
    axes_labelcolor: str = "black"
    tick_color: str = "black"
    figure_dpi: int = 110
    savefig_dpi: int = 300

# Paramètres matplotlib appliqués par `Theme.apply()`.
THEME_RC_OVERRIDES = {
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "svg.fonttype": "none",
    "figure.facecolor": "#FFFCF2",
    "axes.facecolor": "#FFFCF2",
}


def configure_theme(cfg: ThemeConfig) -> None:
    """Applique une configuration **externe** au thème.

    Cette fonction met à jour les couleurs et palettes de :class:`Theme`
    et prépare les paramètres matplotlib (``rcParams``) pour le fond,
    les couleurs de texte et les résolutions.

    Paramètres
    ----------
    cfg : ThemeConfig
        Instance contenant l'ensemble des options de thème.
    """
    # Configure la palette dans la classe
    Theme.configure(
        primary=cfg.primary,
        secondary=cfg.secondary,
        tertiary=cfg.tertiary,
        primary_variants=cfg.primary_variants,
        secondary_variants=cfg.secondary_variants,
        tertiary_variants=cfg.tertiary_variants,
        sequential_light=cfg.sequential_light,
        sequential_dark=cfg.sequential_dark,
        light_amount=cfg.light_amount,
        dark_amount=cfg.dark_amount,
    )
    Theme.BACKGROUND = cfg.background
    # Prépare les overrides rcParams
    THEME_RC_OVERRIDES.update({
        "text.color": cfg.text_color,
        "axes.labelcolor": cfg.axes_labelcolor,
        "xtick.color": cfg.tick_color,
        "ytick.color": cfg.tick_color,
        "figure.dpi": cfg.figure_dpi,
        "savefig.dpi": cfg.savefig_dpi,
        # On conserve ces valeurs par défaut qui ne dépendent pas de la couleur
        "savefig.bbox": "tight",
        "svg.fonttype": "none",
        "figure.facecolor": cfg.background,
        "axes.facecolor": cfg.background,
    })


def _config_from_mapping(data: Mapping[str, Any]) -> ThemeConfig:
    """Convertit un mapping arbitraire en :class:`ThemeConfig`."""
    allowed_fields = {f.name for f in fields(ThemeConfig)}
    unknown_keys = set(data) - allowed_fields
    if unknown_keys:
        raise ValueError(
            "Clés inconnues dans la configuration du thème: "
            + ", ".join(sorted(unknown_keys))
        )
    filtered = {k: data[k] for k in allowed_fields if k in data}
    return ThemeConfig(**filtered)


def load_brand(path: Union[str, Path]) -> ThemeConfig:
    """Charge un fichier YAML et retourne une configuration de thème.

    Le fichier doit contenir des clés correspondant aux attributs de
    :class:`ThemeConfig`. Les valeurs absentes conservent les valeurs
    par défaut de la dataclass.
    """

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dépendance optionnelle
        raise RuntimeError(
            "PyYAML est requis pour charger une charte graphique YAML. "
            "Installez le paquet 'pyyaml'."
        ) from exc

    file_path = Path(path).expanduser()
    if not file_path.exists():
        raise FileNotFoundError(f"Fichier YAML introuvable: {file_path}")

    with file_path.open("r", encoding="utf-8") as handle:
        content = yaml.safe_load(handle) or {}

    if not isinstance(content, Mapping):
        raise ValueError(
            "Le contenu du YAML doit être un mapping clé/valeur (dict)."
        )

    return _config_from_mapping(content)


def configure_brand(path: Union[str, Path]) -> ThemeConfig:
    """Charge un fichier YAML puis applique la configuration obtenue."""

    cfg = load_brand(path)
    configure_theme(cfg)
    return cfg


class Theme:
    """Thème graphique pour Matplotlib et Seaborn.

    La classe fournit :
    - des couleurs principales et une palette qualitative étendue ;
    - des cartes de couleurs (séquentielles et divergentes) ;
    - une méthode :meth:`apply` pour appliquer le thème globalement ;
    - des méthodes de démonstration pour un aperçu rapide.

    Les couleurs ne sont pas figées : utilisez
    :func:`configure_theme` pour injecter une configuration externe.
    """

    # --- Couleurs principales (configurables via Theme.configure) ---
    # Valeurs par défaut (seront écrasées par configure())
    PRIMARY: str = "#7451EB"    # violet (chaud)
    SECONDARY: str = "#EE8273"  # corail (chaud)
    TERTIARY: str = "#A6BD63"   # vert (froid)
    BACKGROUND: str = "#FFFCF2"

    PALETTE: List[str] = ["#7451EB", "#EE8273", "#A6BD63"]

    @classmethod
    def base_palette(cls) -> List[str]:
        """Retourne la palette fondamentale (PRIMARY, SECONDARY, TERTIARY)."""
        return [cls.PRIMARY, cls.SECONDARY, cls.TERTIARY]

    # Variantes (palette qualitative) – configurables
    _PRIMARY_VARIANTS: List[str] = ["#9D7EF0", "#4B25D6"]
    _SECONDARY_VARIANTS: List[str] = ["#F3A093", "#D95848"]
    _TERTIARY_VARIANTS: List[str] = ["#BDD681", "#7E923F"]

    # Colormaps séquentiels (clair -> couleur -> foncé) – configurables
    _SEQUENTIALS = {
        "primary":   ["#f3f0fd", PRIMARY,   "#2f1577"],
        "secondary": ["#fdecea", SECONDARY, "#8b3025"],
        "tertiary":  ["#f6faec", TERTIARY,  "#4b5c27"],
    }

    _NAMES = {"primary", "secondary", "tertiary"}

    # --------- Configuration dynamique ---------
    @staticmethod
    def _to_rgb(color: str):
        return np.array(mcolors.to_rgb(color))

    @classmethod
    def _tint(cls, color: str, amount: float = 0.85) -> str:
        """Retourne une version éclaircie de ``color``.

        Le mélange avec le blanc est contrôlé par ``amount`` (0..1).
        """
        c = cls._to_rgb(color)
        white = np.array([1.0, 1.0, 1.0])
        mixed = (1 - amount) * c + amount * white
        return mcolors.to_hex(mixed) # type: ignore

    @classmethod
    def _shade(cls, color: str, amount: float = 0.65) -> str:
        """Retourne une version assombrie de ``color``.

        Le mélange avec le noir est contrôlé par ``amount`` (0..1).
        """
        c = cls._to_rgb(color)
        black = np.array([0.0, 0.0, 0.0])
        mixed = (1 - amount) * c + amount * black
        return mcolors.to_hex(mixed) # type: ignore

    @classmethod
    def _compute_sequentials(
        cls,
        primary: str,
        secondary: str,
        tertiary: str,
        light_overrides: Optional[dict] = None,
        dark_overrides: Optional[dict] = None,
        light_amount: float = 0.85,
        dark_amount: float = 0.65,
    ) -> dict:
        """Construit les stops [clair, milieu, foncé] de chaque couleur.

        Paramètres
        ----------
        primary, secondary, tertiary : str
            Couleurs principales en hex.
        light_overrides, dark_overrides : dict | None
            Remplacements explicites pour les teintes claires/foncées.
        light_amount, dark_amount : float
            Coefficients de mélange quand aucun remplacement n'est fourni.

        Retour
        ------
        dict
            Dictionnaire {nom: [clair, milieu, foncé]}.
        """
        light_overrides = light_overrides or {}
        dark_overrides = dark_overrides or {}
        base = {
            "primary": primary,
            "secondary": secondary,
            "tertiary": tertiary,
        }
        seq = {}
        for k, mid in base.items():
            light = light_overrides.get(k) or cls._tint(mid, amount=light_amount)
            dark = dark_overrides.get(k) or cls._shade(mid, amount=dark_amount)
            seq[k] = [light, mid, dark]
        return seq

    @classmethod
    def configure(
        cls,
        *,
        primary: Optional[str] = None,
        secondary: Optional[str] = None,
        tertiary: Optional[str] = None,
        primary_variants: Optional[List[str]] = None,
        secondary_variants: Optional[List[str]] = None,
        tertiary_variants: Optional[List[str]] = None,
        sequential_light: Optional[dict] = None,
        sequential_dark: Optional[dict] = None,
        light_amount: float = 0.85,
        dark_amount: float = 0.65,
    ) -> None:
        """Met à jour dynamiquement les couleurs et colormaps de la classe.

        Exemples
        --------
        >>> Theme.configure(primary="#0072CE", secondary="#FF6A00")
        >>> Theme.configure(
        ...     primary="#1f77b4",
        ...     sequential_light={"primary": "#eef5fb"},
        ...     sequential_dark={"primary": "#0b3050"},
        ... )

        Paramètres
        ----------
        primary, secondary, tertiary : str | None
            Couleurs principales.
        primary_variants, secondary_variants, tertiary_variants : list[str] | None
            Variantes qualitatives.
        sequential_light, sequential_dark : dict | None
            Remplacements pour les teintes claires/foncées.
        light_amount, dark_amount : float
            Coefficients de mélange par défaut.
        """
        if primary:
            cls.PRIMARY = primary
        if secondary:
            cls.SECONDARY = secondary
        if tertiary:
            cls.TERTIARY = tertiary

        if primary_variants is not None:
            cls._PRIMARY_VARIANTS = primary_variants
        if secondary_variants is not None:
            cls._SECONDARY_VARIANTS = secondary_variants
        if tertiary_variants is not None:
            cls._TERTIARY_VARIANTS = tertiary_variants

        # Recalcule les rampes séquentielles (avec overrides éventuels)
        cls._SEQUENTIALS = cls._compute_sequentials(
            cls.PRIMARY,
            cls.SECONDARY,
            cls.TERTIARY,
            light_overrides=sequential_light,
            dark_overrides=sequential_dark,
            light_amount=light_amount,
            dark_amount=dark_amount,
        )
        cls.PALETTE = cls.base_palette()

    # --------- helpers internes ---------
    @classmethod
    def _get_seq(cls, key: str) -> List[str]:
        """Retourne la rampe séquentielle associée à ``key``.

        Déclenche ``ValueError`` si la clé est inconnue.
        """
        key = key.lower()
        if key not in cls._NAMES:
            raise ValueError(f"Couleur inconnue: {key}. Choisir parmi {sorted(cls._NAMES)}.")
        return cls._SEQUENTIALS[key]

    @staticmethod
    def _from_list(name: str, colors: List[str]) -> mcolors.LinearSegmentedColormap:
        """Crée un ``LinearSegmentedColormap`` à partir d'une liste.
        """
        return mcolors.LinearSegmentedColormap.from_list(name, colors)

    @classmethod
    def _make_diverging(
        cls,
        start_key: str,
        end_key: str,
        *,
        center: Optional[str] = None,
        strong_ends: bool = True,
        blend_center: bool = False,
        blend_ratio: float = 0.5,
    ) -> Tuple[str, List[str]]:
        """Construit un colormap divergent à partir de deux rampes.

        Stops générés :
        ``[dark_start?, start_mid, center, end_mid, dark_end?]``
        """
        s_seq = cls._get_seq(start_key)  # [light, mid, dark]
        e_seq = cls._get_seq(end_key)    # [light, mid, dark]

        if blend_center:
            center_color = mix_colors(s_seq[1], e_seq[1], ratio=blend_ratio)
        else:
            center_color = center or "#f7f7f7"

        colors: List[str] = []
        if strong_ends:
            colors.append(s_seq[2])      # dark start
        colors.append(s_seq[1])          # start mid
        colors.append(center_color)      # centre neutre ou mélange
        colors.append(e_seq[1])          # end mid
        if strong_ends:
            colors.append(e_seq[2])      # dark end

        name = f"ocr_div_{start_key}_{end_key}"
        return name, colors

    # --------- API publique ---------
    @classmethod
    def colormap(
        cls,
        mode: Literal["primary", "secondary", "tertiary", "sequential", "diverging"] = "primary",
        *,
        start: Optional[Literal["primary", "secondary", "tertiary"]] = None,
        end: Optional[Literal["primary", "secondary", "tertiary"]] = None,
        reverse: bool = False,
        as_cmap: bool = True,
        center: Optional[str] = None,
        blend_center: bool = False,
        blend_ratio: float = 0.5,
        strong_ends: bool = True,
    ):
        """Retourne un colormap Matplotlib ou la liste des stops.

        Utilisation
        -----------
        Séquentiel autour d'une couleur :
            colormap("primary")
            colormap("sequential", start="primary")
        Divergent entre deux couleurs :
            colormap("diverging", start="primary", end="tertiary")

        Paramètres
        ----------
        mode : {"primary", "secondary", "tertiary", "sequential", "diverging"}
            Type de colormap souhaité.
        start, end : {"primary", "secondary", "tertiary"} | None
            Couleurs de départ/arrivée (suivant le mode).
        reverse : bool
            Inverse l'ordre des couleurs.
        as_cmap : bool
            Si ``True``, retourne un objet ``Colormap`` ; sinon la liste
            des valeurs hexadécimales.
        center : str | None
            Couleur centrale explicite (hexadécimal). Ignorée si ``blend_center`` vaut ``True``.
        blend_center : bool
            Mélange automatiquement les teintes ``start`` et ``end`` pour générer la couleur centrale.
        blend_ratio : float
            Ratio de mélange (0..1) appliqué quand ``blend_center`` est activé.
        strong_ends : bool
            Ajoute les teintes foncées des rampes aux extrémités du colormap divergent.
        """
        # Alias pour compat : "primary"/"secondary"/"tertiary" => séquentiel
        if mode in {"primary", "secondary", "tertiary"}:
            seq = cls._get_seq(mode)
            colors = list(reversed(seq)) if reverse else seq
            return cls._from_list(f"ocr_{mode}", colors) if as_cmap else colors

        #mode = mode.lower()
        if mode == "sequential":
            key = (start or "primary").lower()
            seq = cls._get_seq(key)
            colors = list(reversed(seq)) if reverse else seq
            return cls._from_list(f"ocr_seq_{key}", colors) if as_cmap else colors

        if mode == "diverging":
            if not start or not end:
                raise ValueError("Pour un colormap diverging, fournir start=... et end=...")
            #start = start.lower()
            #end = end.lower()
            if start not in cls._NAMES or end not in cls._NAMES:
                raise ValueError(f"start/end doivent être dans {sorted(cls._NAMES)}.")
            name, colors = cls._make_diverging(
                start,
                end,
                center=center,
                strong_ends=strong_ends,
                blend_center=blend_center,
                blend_ratio=blend_ratio,
            )
            if reverse:
                colors = list(reversed(colors))
            return cls._from_list(name, colors) if as_cmap else colors

        raise ValueError("mode inconnu. Utiliser 'primary'/'secondary'/'tertiary' ou 'sequential'/'diverging'.")

    @classmethod
    def apply(cls, *, context: str = "notebook", style: str = "white") -> List[str]:
        """Applique le thème global Seaborn/Matplotlib.

        Retourne la palette qualitative étendue utilisée par Seaborn.
        """
        pal = cls.extended_palette()
        sns.set_theme(context=context, style=style, palette=pal) # type: ignore
        plt.rcParams.update(THEME_RC_OVERRIDES)
        return pal

    @classmethod
    def extended_palette(cls) -> List[str]:
        """Retourne la palette qualitative étendue.

        Utile pour des graphiques multi‑séries (barres, lignes, etc.).
        """
        return [
            cls.PRIMARY, *cls._PRIMARY_VARIANTS,
            cls.SECONDARY, *cls._SECONDARY_VARIANTS,
            cls.TERTIARY, *cls._TERTIARY_VARIANTS,
        ]

    # --------- Démos appelables ---------
    @staticmethod
    def _demo_field(n: int = 300):
        """Génère un champ 2D lisse destiné à ``imshow``."""
        x = np.linspace(-3, 3, n)
        y = np.linspace(-3, 3, n)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(X) * np.cos(Y)
        return X, Y, Z

    @staticmethod
    def _demo_matrix(shape: Tuple[int, int] = (10, 12), seed: int = 0):
        """Génère une matrice aléatoire pour des heatmaps reproductibles."""
        rng = np.random.default_rng(seed)
        return rng.standard_normal(shape)

    @classmethod
    def demo_imshow_sequential(
        cls,
        *,
        start: Literal["primary", "secondary", "tertiary"] = "primary",
        reverse: bool = False,
        with_colorbar: bool = True,
        title: Optional[str] = None,
        apply_theme: bool = False,
    ) -> None:
        """Affiche une démo ``imshow`` avec un colormap séquentiel.

        Exemple
        -------
        >>> Theme.demo_imshow_sequential(start="tertiary", reverse=True)
        """
        if apply_theme:
            cls.apply()
        _, _, Z = cls._demo_field()
        cmap = cls.colormap("sequential", start=start, reverse=reverse)
        plt.imshow(Z, cmap=cmap, origin="lower") # type: ignore
        direction = "foncé → clair" if reverse else "clair → foncé"
        plt.title(title or f"Séquentiel {start.upper()} ({direction})")
        if with_colorbar:
            plt.colorbar()
        plt.show()

    @classmethod
    def demo_imshow_diverging(
        cls,
        *,
        start: Literal["primary", "secondary", "tertiary"] = "primary",
        end: Literal["primary", "secondary", "tertiary"] = "secondary",
        reverse: bool = False,
        with_colorbar: bool = True,
        title: Optional[str] = None,
        apply_theme: bool = False,
        center: Optional[str] = None,
        blend_center: bool = False,
        blend_ratio: float = 0.5,
        strong_ends: bool = True,
    ) -> None:
        """Affiche une démo ``imshow`` avec un colormap divergent.

        Exemple
        -------
        >>> Theme.demo_imshow_diverging(start="primary", end="secondary")
        """
        if apply_theme:
            cls.apply()
        _, _, Z = cls._demo_field()
        cmap = cls.colormap(
            "diverging",
            start=start,
            end=end,
            reverse=reverse,
            center=center,
            blend_center=blend_center,
            blend_ratio=blend_ratio,
            strong_ends=strong_ends,
        )
        plt.imshow(Z, cmap=cmap, origin="lower") # type: ignore
        plt.title(title or f"Diverging {start.upper()} ↔ {end.upper()}")
        if with_colorbar:
            plt.colorbar()
        plt.show()

    @classmethod
    def demo_heatmap_sequential(
        cls,
        *,
        start: Literal["primary", "secondary", "tertiary"] = "primary",
        reverse: bool = False,
        title: Optional[str] = None,
        apply_theme: bool = True,
    ) -> None:
        """Affiche une heatmap Seaborn en mode séquentiel.

        Exemple
        -------
        >>> Theme.demo_heatmap_sequential(start="primary")
        """
        if apply_theme:
            cls.apply()
        data = cls._demo_matrix()
        plt.figure(figsize=(6, 4))
        sns.heatmap(data, cmap=cls.colormap("sequential", start=start, reverse=reverse)) # type: ignore
        direction = "foncé → clair" if reverse else "clair → foncé"
        plt.title(title or f"Heatmap séquentielle - {start.upper()} ({direction})")
        plt.show()

    @classmethod
    def demo_heatmap_diverging(
        cls,
        *,
        start: Literal["primary", "secondary", "tertiary"] = "primary",
        end: Literal["primary", "secondary", "tertiary"] = "tertiary",
        reverse: bool = False,
        title: Optional[str] = None,
        apply_theme: bool = True,
    ) -> None:
        """Affiche une heatmap Seaborn en mode divergent.

        Exemple
        -------
        >>> Theme.demo_heatmap_diverging(start="primary", end="tertiary")
        """
        if apply_theme:
            cls.apply()
        data = cls._demo_matrix()
        plt.figure(figsize=(6, 4))
        sns.heatmap(data, cmap=cls.colormap("diverging", start=start, end=end, reverse=reverse)) # type: ignore
        plt.title(title or f"Heatmap diverging - {start.upper()} ↔ {end.upper()}")
        plt.show()



# API fonctionnelle (compatibilité)

def set_theme():
    """Applique le thème OC et retourne la palette étendue.

    Raccourci rétrocompatible de :meth:`Theme.apply`.
    """
    return Theme.apply()


def set_colormap(
    mode: Literal["primary", "secondary", "tertiary", "sequential", "diverging"] = "primary",
    *,
    start: Optional[Literal["primary", "secondary", "tertiary"]] = None,
    end: Optional[Literal["primary", "secondary", "tertiary"]] = None,
    reverse: bool = False,
    as_cmap: bool = True,
):
    """Raccourci pour obtenir un colormap OC.

    Voir :meth:`Theme.colormap` pour le détail des paramètres.
    """
    return Theme.colormap(mode, start=start, end=end, reverse=reverse, as_cmap=as_cmap)

# Configuration par défaut (externe à la classe)
_default_cfg = ThemeConfig(
    primary="#7451EB",
    secondary="#EE8273",
    tertiary="#A6BD63",
    background="#FFFCF2",
    primary_variants=["#9D7EF0", "#4B25D6"],
    secondary_variants=["#F3A093", "#D95848"],
    tertiary_variants=["#BDD681", "#7E923F"],
    sequential_light={
        "primary": "#f3f0fd",
        "secondary": "#fdecea",
        "tertiary": "#f6faec",
    },
    sequential_dark={
        "primary": "#2f1577",
        "secondary": "#8b3025",
        "tertiary": "#4b5c27",
    },
    text_color="black",
)
configure_theme(_default_cfg)
def mix_colors(color1: str, color2: str, ratio: float = 0.5) -> str:
    """Mélange deux couleurs hexadécimales selon ``ratio`` (0-1)."""
    rgb1 = np.array(mcolors.to_rgb(color1))
    rgb2 = np.array(mcolors.to_rgb(color2))
    mixed = (1 - ratio) * rgb1 + ratio * rgb2
    return mcolors.to_hex(mixed) # type: ignore


def make_diverging_cmap(
    primary: str,
    secondary: str,
    name: str = "custom_diverging",
    ratio: float = 0.5,
):
    """Crée un colormap divergent simple (primary → mix → secondary)."""
    mid = mix_colors(primary, secondary, ratio=ratio)
    return mcolors.LinearSegmentedColormap.from_list(name, [primary, mid, secondary])
