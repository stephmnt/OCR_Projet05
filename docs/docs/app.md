# app.py

app.py est l’API Gradio “online” qui expose les prédictions en rechargeant les artefacts produits. Elle n’a aucune raison de relancer l’entraînement, sinon la Space ferait un fit complet à chaque démarrage, ce qui serait coûteux et lent.

::: app.py 