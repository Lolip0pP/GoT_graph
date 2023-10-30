# GoT_graph
A short research on 'Game Of Thrones' characters' links (with Python)
_(описание по-русски ниже)_

The original data is from github.com/mathbeveridge/gameofthrones and represents the Game of Thrones season 1 character interaction network, which was created by analysing fan-generated scripts from genius.com/artists/Game-of-thrones. Characters are connected by (undirected) edges weighted by the number and nature of interactions (whether they mention each other, appear in the same scene, are familiar, etc.). The vertices have a binary characteristic, gender (this column was added by me).

The programme 
  -writes basic information about the graph; 
  -models, as far as possible, similar graphs using three probabilistic graph models and counts their characteristics; 
  -calculates the centrality of vertices according to 5 different metrics;
  -calculates homophily and assortativity metrics;
  -defines clusters in a graph;
  -solves the classification problem using gradient bousting;
  -**visualises** all of this.

The visualisation is interactive, it's done via creating html-files (in the folder where the Python script itself is saved) and it's pretty computer intensive.  It runs at the user's choice at the very end of the code.

* * *

Оригинальные данные взяты с сайта github.com/mathbeveridge/gameofthrones и представляют собой сеть взаимодействия персонажей 1 сезона "Игры престолов", которая создана путем анализа сгенерированных фанатами сценариев из genius.com/artists/Game-of-thrones. Персонажи соединены (ненаправленными) ребрами, взвешенными по количеству и характеру взаимодействий (упоминают ли они друг о друге, появляются ли в одной сцене, знакомы ли и т.п.). У вершин есть бинарная характеристика – пол (самостоятельно доразмеченный столбец).

Программа 
  -считает основную информацию о графе; 
  -моделирует, насколько это возможно, похожие графы с помощью трёх вероятностных графовых моделей и считает их характеристики; 
  -подсчитывает центральности вершин по 5 различным метрикам;
  -считает показатели гомофилии и ассортативности;
  -определяет кластеры в графе;
  -решает задачу классификации с помощью градиентного бустинга;
  -**визуализирует** всё это.

Визуализация интерактивная, выполняется через создание html-файлов (в ту папку, где сохранён сам Python-скрипт) и она довольно сильно нагружает компьютер.  Запускается по выбору пользователя в самом конце кода.

Код сильно закомментирован, так что, надеюсь, читать его будет легко.
