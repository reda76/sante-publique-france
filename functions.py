import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram


def display_circles(pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None):
    for d1, d2 in axis_ranks: # On affiche les 3 premiers plans factoriels, donc les 6 premières composantes
        if d2 < n_comp:

            # initialisation de la figure
            fig, ax = plt.subplots(figsize=(7,6))

            # détermination des limites du graphique
            if lims is not None :
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 30 :
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
            else :
                xmin, xmax, ymin, ymax = min(pcs[d1,:]), max(pcs[d1,:]), min(pcs[d2,:]), max(pcs[d2,:])

            # affichage des flèches
            # s'il y a plus de 30 flèches, on n'affiche pas le triangle à leur extrémité
            if pcs.shape[1] < 30 :
                plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                   pcs[d1,:], pcs[d2,:], 
                   angles='xy', scale_units='xy', scale=1, color="grey")
                # (voir la doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
            else:
                lines = [[[0,0],[x,y]] for x,y in pcs[[d1,d2]].T]
                ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))
            
            # affichage des noms des variables  
            if labels is not None:  
                for i,(x, y) in enumerate(pcs[[d1,d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax :
                        plt.text(x, y, labels[i], fontsize='14', ha='center', va='center', rotation=label_rotation, color="blue", alpha=0.5)
            
            # affichage du cercle
            circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='b')
            plt.gca().add_artist(circle)

            # définition des limites du graphique
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
        
            # affichage des lignes horizontales et verticales
            plt.plot([-1, 1], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-1, 1], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Cercle des corrélations (F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)
        
def display_factorial_planes(X_projected, n_comp, pca, axis_ranks, labels=None, alpha=1, illustrative_var=None):
    for d1,d2 in axis_ranks:
        if d2 < n_comp:
 
            # initialisation de la figure       
            fig = plt.figure(figsize=(7,6))
        
            # affichage des points
            if illustrative_var is None:
                plt.scatter(X_projected[:, d1], X_projected[:, d2], alpha=alpha)
            else:
                illustrative_var = np.array(illustrative_var)
                for value in np.unique(illustrative_var):
                    selected = np.where(illustrative_var == value)
                    plt.scatter(X_projected[selected, d1], X_projected[selected, d2], alpha=alpha, label=value)
                plt.legend()

            # affichage des labels des points
            if labels is not None:
                for i,(x,y) in enumerate(X_projected[:,[d1,d2]]):
                    plt.text(x, y, labels[i],
                              fontsize='14', ha='center',va='center') 
                
            # détermination des limites du graphique
            boundary = np.max(np.abs(X_projected[:, [d1,d2]])) * 1.1
            plt.xlim([-boundary,boundary])
            plt.ylim([-boundary,boundary])
        
            # affichage des lignes horizontales et verticales
            plt.plot([-100, 100], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-100, 100], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Projection des individus (sur F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)

def display_scree_plot(pca):
    scree = pca.explained_variance_ratio_*100
    plt.bar(np.arange(len(scree))+1, scree)
    plt.plot(np.arange(len(scree))+1, scree.cumsum(),c="red",marker='o')
    plt.xlabel("rang de l'axe d'inertie")
    plt.ylabel("pourcentage d'inertie")
    plt.title("Eboulis des valeurs propres")
    plt.show(block=False)

def plot_dendrogram(Z, names):
    plt.figure(figsize=(10,25))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('distance')
    dendrogram(
        Z,
        labels = names,
        orientation = "left",
    )
    plt.show()

# Afficher un tableau des pourcentages des valeurs NaN pour chaque colonne
def tableau_pourcentage_NaN(dataframe):
    pourcentage_valeur_naan = pd.DataFrame(
        dataframe.isna().mean().round(4) * 100,
        columns=['Données manquantes en %'
                 ]).sort_values(by='Données manquantes en %', ascending=False)
    return (pourcentage_valeur_naan)

def graph_bar(df_pourcentage, column, titre, xlabel, ylabel):

    df_graphique = df_pourcentage.loc[:, [column, 'Nombre']].set_index(
        column)['Nombre'].copy()

    plt.figure(figsize=(15, 10))

    sns.set(style="whitegrid")
    g = sns.barplot(x=df_graphique, y=df_graphique.index, orient='h')

    plt.title(titre, fontsize=20)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)

    # Afficher le nombre à droite du graphique à barres
    for i, v in enumerate(df_graphique.values.tolist()):
        g.text(v + 3, i + .25, str(v), color='black', fontweight='normal')

def graph_multivarie(column1, column2, title, df):
    df_selection = df[[column1, column2]]
    df_selection_top = df_selection.dropna()

    df_selection_pourcentage = df_selection_top.groupby([
        column2
    ]).size().reset_index(name='Nombre').sort_values('Nombre',
                                                     ascending=False).head(50)
    rows = df_selection_top.shape[0]
    df_selection_pourcentage[
        'Pourcentage'] = df_selection_pourcentage['Nombre'] / rows * 100

    dict_selection_pourcentage = {}

    liste_selection_50 = df_selection_pourcentage[column2].tolist()

    # Nous plaçons les valeurs de la colonne en clé et leurs moyennes en valeurs 
    # dans le dictionnaire 'dict_selection_pourcentage'
    for i in liste_selection_50:
        a = df_selection.loc[df_selection[column2] == i]
        a = a[column1].mean()
        dict_selection_pourcentage[i] = a

    # Tri les valeurs du dictionnaire en ordre croissant
    liste_selection = sorted(dict_selection_pourcentage.items(),
                             key=lambda x: x[1])

    # Top 10 des pays ayant les produits avec le meilleur nutriscore en moyenne
    liste_selection = liste_selection[:10]

    dict_moyenne_selection_10 = {}

    for i in liste_selection:
        dict_moyenne_selection_10[i[0]] = i[1]

    plt.figure(figsize=(15, 10), dpi=80)

    ind = np.arange(len(dict_moyenne_selection_10))
    palette = sns.color_palette("husl", len(dict_moyenne_selection_10))

    plt.bar(ind, list(dict_moyenne_selection_10.values()), color=palette)
    plt.xticks(ind, list(dict_moyenne_selection_10.keys()))
    plt.title(title)
    plt.show()

    return dict_moyenne_selection_10

