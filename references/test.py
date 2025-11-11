class Greeter:
    """Classe permettant de saluer un utilisateur.

    Cette classe gère la génération de messages de bienvenue personnalisés.
    Elle illustre l'utilisation de docstrings respectant les conventions
    PEP 8 et PEP 257, dans un format compatible avec mkdocstrings.
    """

    def __init__(self, name: str = "Inconnu"):
        """Initialise le greeter avec un nom d'utilisateur.

        Args:
            name (str): Le nom de la personne à saluer. Par défaut "Inconnu".
        """
        self.name = name

    def say_hello(self) -> str:
        """Retourne un message de salutation.

        Returns:
            str: Un message de type "Bonjour, <nom> !".
        """
        return f"Bonjour, {self.name} !"

    def say_goodbye(self) -> str:
        """Retourne un message d'au revoir.

        Returns:
            str: Un message de type "Au revoir, <nom> !".
        """
        return f"Au revoir, {self.name} !"
