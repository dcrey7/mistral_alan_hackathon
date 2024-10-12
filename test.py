import pandas as pd

# Ouvrir le fichier CSV et l'enregistrer dans un DataFrame
df = pd.read_csv("/home/dorian/Documents/code/mistral_alan_hackathon/questions.csv")

# Afficher les premières lignes du DataFrame pour vérifier le contenu
print(df.head())


# Ouvrir un fichier en mode écriture pour enregistrer le markdown
with open("/home/dorian/Documents/code/mistral_alan_hackathon/questions.md", "w") as f:
    for index, row in df.iterrows():
        f.write(f"### Question {index + 1}\n")
        f.write(f"{row['question']}\n")
        f.write("- " + row["answer_A"] + "\n")
        f.write("- " + row["answer_B"] + "\n")
        f.write("- " + row["answer_C"] + "\n")
        f.write("- " + row["answer_D"] + "\n")
        f.write("- " + row["answer_E"] + "\n")
        f.write("\n")
        # Afficher un message de confirmation
        print("Le fichier markdown a été généré avec succès.")
