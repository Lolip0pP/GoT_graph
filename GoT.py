import networkx as nx # для работы с графами
import pandas as pd # для работы с данными
import numpy as np # куда же без математики
from pyvis.network import Network # для визуализации
from networkx.algorithms import tree # остовное дерево
from networkx.algorithms import community # для кластеризации
import catboost as cat # для построения модели
from sklearn.model_selection import train_test_split # для разделения выборки на train и test

'''
# Ручная доразметка данных: меняем значение в столбце Label на 1, если персонаж мужского пола,
# и 0, если персонаж женского пола (даже для драконов разметим, не страшно)
# Offtop question: бывают ли драконы-женщины?
# Offtop answer: если верить "Шреку", бывают.

import pandas as pd
nodes = pd.read_csv('got-s1-nodes.csv')
for i in range(len(nodes)):
    nodes['Label'][i] = input(nodes['Id'][i] + ' ')
nodes.to_csv('got-s1-nodes.csv', index=False)
'''

# Вспомогательная функция (отвечает за округление того, что выводится)
def rou(x):
    round_or_not = True # если True, то округляем всё, что выводим
    if round_or_not: return round(x, 4) # число можно менять, чтобы было больше/меньше знаков после запятой
    else: return x

# Читаем данные и запихиваем их в pandas-таблицу
edges = pd.read_csv('got-s1-edges.csv')
# Если очень хочется посмотреть, как примерно выглядит таблица, вот команда:
#print(edges.head())
nodes = pd.read_csv('got-s1-nodes.csv')

# Создаём граф, гордо именуемый G
G = nx.from_pandas_edgelist(edges,'Source', 'Target', edge_attr=True, create_using=nx.Graph())
# Считаем плотность (потом в расчётах понадобится)
p = nx.density(G)
# Считаем количество вершин (в расчётах понадобится прямо сейчас)
l = G.number_of_nodes()

# Раскрашиваем вершины: женщин в красный, мужчин в зелёный
color = ['#FF0000']*l
men = 0
for i in range(l): 
    if nodes['Label'][i] == 1: 
        color[i] = '#00FF00'
        men += 1
#Создаём список пар персонаж-цвет (тоже потом понадобится)
colors_of_nodes = [(nodes['Id'][i], color[i]) for i in range(len(color))]

# Основная информация о графе
av_degree = 0
for i in nx.degree(G):
    av_degree += i[1] /  l
print('Персонажей-мужчин ', men, ', персонажей-женщин ', l-men, '.\n', sep = '')
print('Количество вершин: ', l)
print('Количество рёбер: ', G.number_of_edges())
print('Плотность графа: ', rou(p))
print('Средняя степень вершины в графе: ', rou(av_degree))
print('Диаметр графа: ', nx.diameter(G))
print('Средний кратчайший путь между вершинами: ', rou(nx.average_shortest_path_length(G)))
print('Транзитивность графа: ', rou(nx.transitivity(G)))
print()

# Генерируем n графов и считаем их средние величины (и так для трёх моделей)
n = 100 # количество нагенерированных графов

# Эрдеш-Реньи
av_diam = 0
av_path = 0
av_trans = 0
unconnected = 0 # количество графов, которые получатся несвязными
for i in range(n):
    G_er = nx.erdos_renyi_graph(l, p, directed=False)
    if nx.is_connected(G_er):
        av_diam += nx.diameter(G_er) / n
        av_path += nx.average_shortest_path_length(G_er) / n
    else:
        unconnected += 1
    av_trans += nx.transitivity(G_er) / n
print('Для модели Эрдеша-Реньи (количество генераций = ', n, '):',
      '\n\tколичество несвязных графов = ', unconnected,
      '\n\tсредний диаметр = ', rou(av_diam),
      '\n\tсредний средний кратчайший путь = ', rou(av_path),
      '\n\tсредняя транзитивность = ', rou(av_trans), sep = '')
print()

# Уоттс-Строгатц
av_diam = 0
av_path = 0
av_trans = 0
param_p = 0.5 # не знаю, как этот параметр выбирать, тыкнул наобум
for i in range(n):
    G_ws = nx.watts_strogatz_graph(l, int(av_degree), param_p)
    av_diam += nx.diameter(G_ws) / n
    av_path += nx.average_shortest_path_length(G_ws) / n
    av_trans += nx.transitivity(G_ws) / n
print('Для модели Уоттса-Строгатца (количество генераций = ', n, ', p = ', param_p, '):',
      '\n\tсредний диаметр = ', rou(av_diam),
      '\n\tсредний средний кратчайший путь = ', rou(av_path),
      '\n\tсредняя транзитивность = ', rou(av_trans), sep = '')
print()

# Барабаши-Альберт
av_diam = 0
av_path = 0
av_trans = 0
for i in range(n):
    G_ws = nx.barabasi_albert_graph(l, int(av_degree))
    av_diam += nx.diameter(G_ws) / n
    av_path += nx.average_shortest_path_length(G_ws) / n
    av_trans += nx.transitivity(G_ws) / n
print('Для модели Барабаши-Альберт (количество генераций = ', n, '):',
      '\n\tсредний диаметр = ', rou(av_diam),
      '\n\tсредний средний кратчайший путь = ', rou(av_path),
      '\n\tсредняя транзитивность = ', rou(av_trans), sep = '')
print()

def vis_all():
    # Визуализация всего графа: малоинформативная, зато красивая
    # Раскрашиваем вершины
    color = ['#FF0000']*l
    men = 0
    for i in range(l): 
        if nodes['Label'][i] == 1: 
            color[i] = '#00FF00'
            men += 1
    # Создаём объект с гордым названием network, добавляем к нему узлы и рёбра 
    network = Network('700px', '1500px', notebook=True, cdn_resources='remote', bgcolor="#222222", font_color="white") #размеры можно подогнать, если очень хочется
    network.add_nodes(nodes['Id'], color = color)
    # Библиотека pyvis не дружит с библиотекой pandas, поэтому нужно добавить немного магии с приведением типов
    edge_net = []
    for i in range(len(edges['Source'])): #здесь проблемный 3-й аргумент, где веса рёбер, можно его убрать, чтобы веса не отображались на графе, т.к. он и так перегружен
        edge_net.append((edges['Source'][i], edges['Target'][i], 
                         int(np.sqrt(edges['Weight'][i]))
                         ))
    network.add_edges(edge_net)
    # Добавляем слегка откалиброванные настройки физического алгоритма
    network.force_atlas_2based(gravity=-100, central_gravity=0.01, spring_length=100, spring_strength=0.4, damping=0.5, overlap=0)
    # Добавляем возможность менять физику или вообще её выключить
    network.show_buttons(['physics'])
    network.show('network.html', notebook = False)


def vis_strong():
    # Упрощение визуализации
    # Путь №1: оставляем только сильные связи
    network1 = Network('700px', '1500px', notebook=True, cdn_resources='remote', bgcolor="#222222", font_color="white") 
    # Ограничиваем множество рёбер, ставя ограничение по весу
    edge_net = []
    for i in range(len(edges)):
        if edges['Weight'][i] > 20: # это значение можно менять, смотря какого размера граф мы хотим
            edge_net.append((edges['Source'][i], edges['Target'][i]))
    # Немного колдовства, чтобы оставить только вершины из сокращённого множества рёбер
    n = len(edge_net)
    nodes = list(set([edge_net[i][0] for i in range(n)]) | set([edge_net[i][1] for i in range(n)]))
    # Раскрашиваем вершины (тоже без магии не обойдёмся)
    color = ['']*len(nodes)
    j = 0
    for i in range(len(colors_of_nodes)):
        j = colors_of_nodes[i][0]
        if j in nodes: 
            color[nodes.index(j)] = colors_of_nodes[i][1]

    network1.add_nodes(nodes, color = color)
    network1.add_edges(edge_net)
    # Оставляем те же настройки, что и для большого графа, для наглядности
    network1.force_atlas_2based(gravity=-100, central_gravity=0.01, spring_length=100, spring_strength=0.4, damping=0.5, overlap=0)
    network1.show_buttons(['physics'])
    # Здесь можно пронаблюдать кластеризацию
    network1.show('network1.html', notebook = False)


def vis_ost():
    # Путь №2: делаем остовное дерево
    # Как стало видно из предыдущего опыта, у нас стало больше компонент связности. Но что, если мы хотим оставить одну?
    # Для этого делаем остовное дерево (грубо говоря, удаляем все рёбра, не увеличивающие кол-во компонент связности)
    network2 = Network('700px', '1500px', notebook=True, cdn_resources='remote', bgcolor="#222222", font_color="white") 
    # Применяем алгоритм, придуманный за нас умными людьми
    mst = tree.maximum_spanning_edges(G, algorithm='kruskal', data=False)
    # Список рёбер
    edge_net = list(mst)
    # Список вершин (как в прошлый раз)
    n = len(edge_net)
    nodes = list(set([edge_net[i][0] for i in range(n)]) | set([edge_net[i][1] for i in range(n)]))
    # Раскрашиваем вершины (тоже уже было)
    color = ['']*len(nodes)
    j = 0
    for i in range(len(colors_of_nodes)):
        j = colors_of_nodes[i][0]
        if j in nodes: 
            color[nodes.index(j)] = colors_of_nodes[i][1]

    network2.add_nodes(nodes, color = color)
    network2.add_edges(edge_net)
    # Оставляем те же настройки, что и для большого графа, для наглядности
    network2.force_atlas_2based(gravity=-100, central_gravity=0.01, spring_length=100, spring_strength=0.4, damping=0.5, overlap=0)
    network2.show_buttons(['physics'])
    # Здесь можно красиво наблюдать самые центральные вершины
    network2.show('network2.html', notebook = False)


# Подсчёт центральностей

n = 5 # выводятся n самых центральных вершин
print(n, 'самых центральных вершин по... ')
# по степени
characters = sorted(list(nx.degree_centrality(G).items()), key=lambda i: i[1], reverse=True)
print('\tстепени:')
for i in range(n): print('\t\t', characters[i][0], ': ', rou(characters[i][1]), sep = '')
# по собственному значению
characters = sorted(list(nx.eigenvector_centrality(G).items()), key=lambda i: i[1], reverse=True)
print('\n\tсобственному значению:')
for i in range(n): print('\t\t', characters[i][0], ': ', rou(characters[i][1]), sep = '')
# по близости
characters = sorted(list(nx.closeness_centrality(G).items()), key=lambda i: i[1], reverse=True)
print('\n\tблизости:')
for i in range(n): print('\t\t', characters[i][0], ': ', rou(characters[i][1]), sep = '')
# по between-ности
characters = sorted(list(nx.betweenness_centrality(G).items()), key=lambda i: i[1], reverse=True)
print('\n\tbetweenness:')
for i in range(n): print('\t\t', characters[i][0], ': ', rou(characters[i][1]), sep = '')
# PageRank centrality
characters = sorted(list(nx.pagerank(G).items()), key=lambda i: i[1], reverse=True)
print('\n\tPageRank:') # центральность, по которой Google ранжирует интернет-страницы
for i in range(n): print('\t\t', characters[i][0], ': ', rou(characters[i][1]), sep = '')
print()

# Считаем dyadicity и heterophilicity
e_11 = 0 # рёбер между вершинами с характеристикой = 1
e_00 = 0 # рёбер между вершинами с характеристикой = 0
e_01 = 0 # рёбер между разнохарактеристичными вершинами
# Делаем магические пассы (я просто не придумал, как подсчитать это проще)
source = list(edges['Source'])
target = list(edges['Target'])
for i in range(len(source)):
    for j in range(len(nodes['Label'])):
        if source[i] == nodes['Id'][j]: source[i] = nodes['Label'][j]
        if target[i] == nodes['Id'][j]: target[i] = nodes['Label'][j]
for i in range(len(source)):
    e_11 += source[i]*target[i]
    if source[i] != target[i]: e_01 += 1
    elif source[i] + target[i] == 0: e_00 += 1
n_c = color.count('#00FF00')
# Вуаля!
dyadicity = e_11/(p*n_c*(n_c-1)/2)
heterophilicity = e_01/(p*n_c*(len(color)-n_c))
print('Dyadicity:', rou(dyadicity), '\nHeterophilicity:', rou(heterophilicity))

# Считаем ассортативности
s = e_11 + e_00 + e_01
e_11 /= s
e_00 /= s
e_01 /= s
r = (e_11+e_00-e_01*e_01)/(1-e_01*e_01) 
print('Ассортативность по степени:', rou(nx.degree_assortativity_coefficient(G)), 
      '\nАссортативность по половому признаку:', rou(r), '\n')

# Кластеризация (спойлер: получилась кластеризация по сюжетным веткам)
print('Средний локальный кластерный коэффициент:', rou(nx.average_clustering(G)))
print('Глобальный кластерный коэффициент:', rou(nx.transitivity(G))) # с триплетами
print()

# Создаём кластеры (несколько итераций алгоритма Гирмана-Ньюмена)
communities_generator = nx.community.girvan_newman(G)
top_level_communities = next(communities_generator)
next_level_communities = next(communities_generator)
# Получилось 3 кластера: весь Вестерос (самый большой, сделаем синим), 
# отдельно Чёрный замок на Стене (связующее звено - Джон, кластер сделаем фиолетовым),
# отдельно Эссос (связующее звено - Дейнерис, кластер сделаем оранжевым).
# Делать больше кластеров нет смысла (экспериментально проверено)

# Раскрашиваем кластеры (т.к. кластеризация у нас получилась по сюжетным веткам, а не по полу, игнорируем нашу бинарную характеристику)
clusters = [0]*l
color = ['#800080']*l
for i in range(l): 
    if nodes['Id'][i] in next_level_communities[0]: 
        color[i] = '#0000FF'
        clusters[i] = 1
    elif nodes['Id'][i] in next_level_communities[1]: 
        color[i] = '#FFA500' 
        clusters[i] = 2
# Создаём список пар персонаж-кластер
clusters_of_nodes = [(nodes['Id'][i], clusters[i]) for i in range(len(color))]

def vis_clust():
    # Визуализируем по-новому раскрашенный граф    
    network_clustered = Network('700px', '1500px', notebook=True, cdn_resources='remote', bgcolor="#222222", font_color="white") #размеры можно подогнать, если очень хочется
    network_clustered.add_nodes(nodes['Id'], color = color)
    # Приводим типы так же, как в самой первой визуализации
    edge_net = []
    for i in range(len(edges['Source'])): #как и раньше, веса на рёбрах можно заигнорировать
        edge_net.append((edges['Source'][i], edges['Target'][i], 
                         int(np.sqrt(edges['Weight'][i]))
                         ))
    network_clustered.add_edges(edge_net)
    # Настройки физики опять не меняем
    network_clustered.force_atlas_2based(gravity=-100, central_gravity=0.01, spring_length=100, spring_strength=0.4, damping=0.5, overlap=0)
    # Добавляем возможность менять физику или вообще её выключить
    network_clustered.show_buttons(['physics'])
    network_clustered.show('network_clustered.html', notebook = False)


# Делаем модель с помощью CatBoost - библиотеки от Яндекса для градиентного бустинга
# Создаём массив переменных, по которым будем обучать модель
centralities = [] # PageRank-центральности вершин
gender = [] # пол персонажей
degrees = [] # степени вершин
av_neighbor = [] # средняя степень соседей
for i in G.nodes():
    centralities.append(nx.pagerank(G).get(i))
    degrees.append(G.degree(i))
    av_neighbor.append(nx.average_neighbor_degree(G).get(i))
    for j in colors_of_nodes:
        if i == j[0]: 
            if j[1] == '#00FF00': gender.append(1)
            else: gender.append(0)
pre_x = [gender, degrees, centralities, av_neighbor]
X = []
for i in range(l): X.append([pre_x[0][i], pre_x[1][i], pre_x[2][i], pre_x[3][i]])

# Создаём массив целевой переменной
y = []
for i in G.nodes():
    for j in clusters_of_nodes:
        if i == j[0]: 
            y.append(j[1])

# Делим выборку на train и test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
eval_dataset = cat.Pool(X_test, y_test) # Pool - это специальный формат данных, который хорошо понимает CatBoost

# Создаём и обучаем модель (заодно рисуем график loss-функции и сохраняем его в файл 'loss.html')
model = cat.CatBoostClassifier(iterations=19, loss_function='MultiClass', use_best_model=True)
model.fit(X_train, y_train, eval_set=eval_dataset, plot_file='loss.html')

# Делаем предсказание вероятности попадания в каждый класс для каждого объекта из тестовой выборки
y_pred = model.predict_proba(X_test)

# Для каждого объекта из тестовой выборки выбираем класс, куда у него наибольшая предсказанная вероятность попасть
preds = []
for i in range(len(y_pred)):
    m = -1
    jj = -1
    for j in range(3):
        if y_pred[i][j] > m:
            m = y_pred[i][j]
            jj = j
    preds.append(jj==y_test[i])
print('Модель предсказала верно ', preds.count(True), ' значений; неверно ' , preds.count(False), ' значений.', sep = '')
print()

print('''Хотите запустить визуализацию?
      0 - нет
      1 - весь граф
      2 - подграф с сильными связями (вес >20)
      3 - остовное дерево
      4 - кластеризованный граф
      5 - все сразу (компьютеру придётся тяжело)
      Чтобы выйти, нажмите 0.''')
while True:
    s = input()
    if s == '1': 
        vis_all()
    elif s == '2': 
        vis_strong()
    elif s == '3': 
        vis_ost()
    elif s == '4': 
        vis_clust()
    elif s == '5': 
        vis_all()
        vis_strong()
        vis_ost()
        vis_clust()
    elif s == '0': 
        print('Мудрое решение!')
        break
    else: print('Не понял, повторите?..')
    
# Спасибо за внимание, надеюсь, мой код принёс вам не очень много страданий.