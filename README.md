# ML_MUSIC
machine learning project using music datasets 

En gros les commandes de base pour ce git ça va être 

git clone [URL du GIT]
-> ca va clone le repertoire qui est en ligne sur votre ordi -> donc tout les fichiers, faites ça au début et au cas ou vous voulez réimporter le projet

git add [les dossiers en questions]
-> une fois que vous avez modifiés les fichiers que vous voulez en gros cest pour les ajouter au prochain commit 

git add -A
-> add tout les dossiers modifiés 

git commit -m [nom du commit]
-> vous allez commit tout les dossiers qui ont été add avec la commande précédente dans le commit 

git push 
-> pour push le commit sur le serveur git 

git pull
-> récuperer l'état du serveur en local sur l'ordi 

git status 
-> affiche l'état du git donc quels fichiers ont été modifiés, add, si il y a un commit ou pas : faites cette commande à chaque fois que vous faites quoi que ce soit histoire d'être sur que ya pas problème 

IMPORTANT: faites dans l'ordre -> MODIFIER/ADD/COMMIT/PULL/PUSH quand vous voulez modifier le git 

si jamais le manuel de git est super utile https://git-scm.com/docs/user-manual

------------------------------------ USEFUL COMMANDS FOR COMPILING ---------------------------------------------

#logistic regression
python main.py --dataset="music" --path_to_data=<where you placed the data folder> --method_name="logistic_regression"

#logistic regression with cross validation
python main.py --dataset="music" --path_to_data=<where you placed the data folder> --method_name="logistic_regression" --use_cross_validation

#ridge regression with 0.1 lambda
python main.py --dataset="music" --path_to_data=<where you placed the data folder> --method_name="ridge_regression" --ridge_regression_lmda =0.1

#ridge regression with 0 lambda (linear regression)
python main.py --dataset="music" --path_to_data=<where you placed the data folder> --method_name="ridge_regression" --ridge_regression_lmda =0

#ridge regression with cross validation
python main.py --dataset="music" --path_to_data=<where you placed the data folder> --method_name="ridge_regression" --use_cross_validation

 
 COMMANDE FINALE POUR LE NN 
 python main.py --dataset="music" --path_to_data=C:\Users\vasarinocode\Desktop\PYTHON\ML\ML_MUSIC\341018_340201_313191_project --method_name="nn" --batch_size=128 --lr=0.1 --max_iters=10

 