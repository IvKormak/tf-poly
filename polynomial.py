import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from itertools import product, combinations_with_replacement
from copy import deepcopy
import sympy

def binomial(x):
    """
    Принимает на вход массив чисел (1, 2), возвращает бином на их основе
    """
    if isinstance(x, (list, tuple)):
        x = np.array(x, dtype=np.float32)
    return np.stack(
        [
            x[0]**2,
            x[0],
            x[0]*x[1],
            x[1],
            x[1]**2,
            np.ones(shape=x[0].shape)
        ], 0)

class Tree:
    """
    Класс для хранения полиномиального древа
    """
    def __init__(self, inputs, restrictions):
        """
        Инициализация входного и первого слоя древа.
        inputs: список строк-меток входов сети
        restrictions: имплицитная информация о весах нейронов первого слоя, (1, 6) массив чисел от 0 до 1
        """
        last_node = inputs.shape[0]-1
        layers = pd.DataFrame({
            'layer_id': np.full(inputs.shape[0], 0),
            'node_id': np.arange(inputs.shape[0]),
            'node': inputs,
            'restrictions': np.tile(np.full(6, np.nan), (inputs.shape[0],1)).tolist(),
            'error': np.full(inputs.shape[0], np.inf)
        })
        combinations = np.array(list(combinations_with_replacement(inputs, 2)))
        
        if restrictions is None:
            restrictions = np.ones(6)
        res_tiled = np.tile(restrictions, (combinations.shape[0],1))
        
        new_layer = {
            'layer_id': np.full(combinations.shape[0], 1),
            'node_id': np.arange(last_node+1, last_node+1+combinations.shape[0]),
            'node': [str(x) for x in np.arange(last_node+1, last_node+1+combinations.shape[0])],
            'restrictions': res_tiled.tolist(),
            'error': np.full(combinations.shape[0], np.inf)
        }

        new_nodes = {
            'node_id': new_layer['node_id'],
            'node': new_layer['node'],
            'left': np.array(list(zip(*combinations))[0]),
            'right': np.array(list(zip(*combinations))[1])
        }
        new_weights = {n:np.random.normal(0,1,6)*restrictions for n in new_layer['node']}
        
        self.connections = pd.DataFrame(new_nodes)
        self.layers = pd.concat((layers, pd.DataFrame(new_layer))).reset_index(drop=True)
        self.top_layer = self.layers.loc[self.layers['layer_id']==1]
        self.weights = new_weights
        
    def new_layer(self, restrictions=None, k=None):
        """
        Инициализация второго и последующего слоёв сети.
        restrictions: имплицитная информация о весах нейронов нового слоя, (1, 6) массив чисел от 0 до 1
        k: количество нейронов, сохраняемое с предыдущего слоя
        """
        if k is None:
            k = self.layers.size
        self.prune(top_k=k)
        connections, layers, weights = self.connections, self.layers, deepcopy(self.weights)
        last_node = layers['node_id'].max()
        last_layer = layers['layer_id'].max()
        inputs = layers.loc[(layers['layer_id']==0),'node']
        combinations = np.array(list(product(inputs, self.top_layer['node'])))
        
        if restrictions is None:
            restrictions = np.ones(6)
        res_tiled = np.tile(restrictions, (combinations.shape[0],1))

        new_layer = {
            'layer_id': np.full(combinations.shape[0], last_layer+1),
            'node_id': np.arange(last_node+1, last_node+1+combinations.shape[0]),
            'node': [str(x) for x in np.arange(last_node+1, last_node+1+combinations.shape[0])],
            'restrictions': res_tiled.tolist(),
            'error': np.full(combinations.shape[0], np.inf)
        }

        new_nodes = {
            'node_id': new_layer['node_id'],
            'node': new_layer['node'],
            'left': np.array(list(zip(*combinations))[0]),
            'right': np.array(list(zip(*combinations))[1])
        }
        new_weights = {n:np.random.normal(0,1,6)*restrictions for n in new_layer['node']}
        
        self.connections = pd.concat((self.connections, pd.DataFrame(new_nodes))).reset_index(drop=True)
        self.layers = pd.concat((self.layers, pd.DataFrame(new_layer))).reset_index(drop=True)
        self.top_layer = self.layers.loc[self.layers['layer_id']==self.layers['layer_id'].max()]
        self.weights.update(new_weights)
    
    def calculate(self, input_values, weights = None):
        """
        Расчёт значения всех узлов сети на данных векторах входных данных
        input_values: словарь векторов входных данных. Ключи словаря - метки входов сети. Все вектора должны быть одной длины
        weights: модифицированные значения весов, для расчётов внутри цикла обучения
        
        Возвращает словарь, в котором каждому узлу сети поставлен в соответствие вектор значений.
        """
        if weights is None:
            weights = self.weights
        conn, layers = self.connections, self.layers
        input_values = {k: np.array(v, dtype=np.float32) for k, v in input_values.items()}

        for _, node, nleft, nright in conn.values:
            left = np.array(input_values[nleft])
            right = np.array(input_values[nright])
            input_values[node] = np.einsum("ij,i->j", binomial((left, right)), weights[node])
        return input_values
    
    def regress(self,
                input_values, 
                true_values, 
                conv_thres=1e-4, 
                iter_thres=1000):
        """
        Оптимизация весов последнего слоя сети при помощи градиентного спуска
        
        input_values: словарь векторов входных данных. Ключи словаря - метки входов сети. Все вектора должны быть одной длины
        true_values: вектор целевых значений аппроксимируемой функции.
        conv_thres: порог схождения. Процесс обучения продолжается, пока относительное изменение среднеквадратичной ошибки не меньше порогового значения.
        iter_thres: макс. количество шагов расчёта.
        """
        conv_thres+=1 
        conn, layers, weights = self.connections, self.layers, deepcopy(self.weights)
        last_layer = self.top_layer
        last_layer_nodes = last_layer['node'].values
        last_layer_restrictions = last_layer['restrictions'].values

        mse = np.ones_like(last_layer_nodes)
        mse_prev = mse*conv_thres+1
        iteration = 0
        
        beta = 0.707 #коэффициент уменьшения шага для цикла поиска оптимального значения шага спуска
        step = np.ones_like(last_layer_nodes)

        values = self.calculate(input_values)
        while (np.abs(mse_prev/mse)>(conv_thres)).any() and iteration < iter_thres:
            v = np.sum(np.array([values[n] for n in last_layer_nodes]), axis=1)
            #print(f"{iteration=}, {mse.min()=}")
            iteration += 1
            calc = 0
            step_optimal = np.ones_like(last_layer_nodes)*-1
            step = step/(beta**2)
            error = np.array([values[node] - true_values for node in last_layer_nodes])
            mse = np.mean(error**2, axis=1)
            grad = np.zeros((last_layer_nodes.shape[0],6))
            for nn, (_, node, nleft, nright) in enumerate(conn[conn['node'].isin(last_layer_nodes)].values):
                grad[nn] = np.einsum("ji,i->j", binomial((values[nleft], values[nright])), error[nn])
                #в теории домножение на вторую производную должно улучшить схождение, но я не получил прироста скорости или качества
                #grad[nn] *= np.einsum("ji->j", binomial((values[nleft], values[nright]))**2)
            mse_n = np.ones_like(last_layer_nodes)
            mse_nn = np.ones_like(last_layer_nodes)
            #цикл подбора оптимального значения шага градиентного спуска
            while ((mse_n <= mse_nn).any() or (mse_n > mse).any()) and (step > 1e-30).all():
                calc += 1
                weights_n = deepcopy(weights)
                step *= beta
                for nn, node in enumerate(last_layer_nodes):
                    weights_n[node] = weights_n[node]-grad[nn]*last_layer_restrictions[nn]*step[nn]
                values_n = self.calculate(input_values, weights=weights_n)
                vn = np.sum(np.array([values_n[n] for n in last_layer_nodes]), axis=1)   
                error_n = np.array([values_n[node] - true_values for node in last_layer_nodes])
                mse_nn = mse_n
                mse_n = np.mean(error_n**2, axis=1)
                step_optimal = mse_n-mse_nn
            values  = values_n
            weights = weights_n
            mse_prev = mse
            mse = mse_n
        layers.loc[layers['node'].isin(last_layer_nodes),'error'] = mse
        
        self.weights = weights
        
    def prune(self, *, root=None, top_k=None):
        """
        Прополка лишних узлов сети.
        Принимает либо root, либо top_k в качестве аргумента
        root: строка-ярлык верхнего узла сети. В сети остаются только узлы, связанные с его входами
        top_k: число k лучших узлов, которые следует оставить в сети
        """
        conn, layers, weights = self.connections, self.layers, self.weights
        
        last_layer = self.layers[self.layers['layer_id']==self.layers['layer_id'].max()]
        
        if root is not None:
            if isinstance(root, str):
                nodes_to_keep = [root]
            else:
                nodes_to_keep = list(root)
                
        elif top_k is not None:
            nodes_to_keep = list(last_layer.sort_values('error').head(top_k)['node'].values)
            
        else:
            raise ValueError(f'Exactly one of `root` or `top_k` should be given, got {"both" if "root" in prune and "top_k" in prune else "none"} instead')
            
        for _, node, nleft, nright in conn.sort_values('node_id', ascending=False).values:
            if node in nodes_to_keep:
                nodes_to_keep += [nleft, nright]
        nodes_to_keep = list(set(nodes_to_keep))

        nweights = {k:v for k, v in weights.items() if k in nodes_to_keep}
        
        self.connections = conn.loc[conn['node'].isin(nodes_to_keep)]
        self.layers = layers.loc[layers['node'].isin(nodes_to_keep)]
        self.top_layer = self.layers.loc[self.layers['layer_id']==self.layers['layer_id'].max()]
        self.weights = nweights
        
    def node_to_equation(self, node=None):
        """
        Возвращает sympy-уравнение, соответствующее узлу с адресом node. 
        node: строка-ярлык узла, для которого требуется рассчитать уравнение. Если не дано, расчитывается для узла с наименьшим значением ошибки
        
        Возвращает sympy-уравнение
        """
        if node is None:
            last_layer = self.layers[self.layers['layer_id']==self.layers['layer_id'].max()]
            node = last_layer.sort_values('error').head(1)['node'].values[0]
            
        conn, layers, weights = self.connections, self.layers, self.weights
        s = {n: sympy.Symbol("№"+n) for n in layers['node'].values}
        inputs = layers.loc[layers['layer_id']==0, 'node'].values
        left, right = conn.loc[conn['node'] == node, ['left', 'right']].values[0]
        w = weights[node]
        node_poly = w[0]*s[left]**2+w[1]*s[left]+w[2]*s[left]*s[right]+w[3]*s[right]+w[4]*s[right]**2+w[5]
        if left not in inputs:
            node_poly = node_poly.subs(s[left], self.node_to_equation(left))
        if right not in inputs:
            node_poly = node_poly.subs(s[right], self.node_to_equation(right))
        return node_poly
    
    def error(self):
        """
        Возвращает наименьшее значение ошибки для нейронов верхнего слоя сети
        """
        return self.layers.sort_values('error').loc[self.layers['layer_id']==self.layers['layer_id'].max(), 'error'].head(1).values[0]
    