## Dane

🔗 Pobierz `planttraits2024.zip` z [Google Drive](https://drive.google.com/...) i wypakuj do folderu `data/`.
Pobierz swój plik API z Kaggle:
Na stronie Kaggle (po zalogowaniu) → kliknij na swój avatar → "Account" → na dole: "Create New API Token"
Plik kaggle.json zostanie pobrany.

Skopiuj plik do folderu:
mkdir ~/.kaggle          # na Linux/macOS
mkdir %USERPROFILE%\.kaggle  # na Windows

Umieść tam kaggle.json i ustaw uprawnienia (Linux):
chmod 600 ~/.kaggle/kaggle.json

Pobierz dane:
kaggle competitions download -c planttraits2024
unzip planttraits2024.zip -d data/
