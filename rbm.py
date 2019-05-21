from __future__ import print_function
import numpy as np

class RBM:
  
  def __init__(self, num_visible, num_hidden):
    self.num_hidden = num_hidden    #Numero de nós ocultos
    self.num_visible = num_visible  #Numero de nós visíveis
    self.debug_print = True


    # A matriz de pesos é iniciada com uma distribuição uniforme entre "low" e "high" 
    # possuindo dimensão nós_visiveis x nós_ocultos (num_visible x num_hidden)
    # Os pesos serão iniciados entre 0 e 0.1 para ser possível variar o desvio padrão.
    np_rng = np.random.RandomState(1234) 

    self.weights = np.asarray(np_rng.uniform(
			low=-0.1 * np.sqrt(6. / (num_hidden + num_visible)),
      high=0.1 * np.sqrt(6. / (num_hidden + num_visible)),
      size=(num_visible, num_hidden))) 


    # Inserindo os pesos na primeira linha e primeira coluna
    self.weights = np.insert(self.weights, 0, 0, axis = 0)
    self.weights = np.insert(self.weights, 0, 0, axis = 1)


  ##### Treinando a maquina #####

  def train(self, data, max_epochs = 1000, learning_rate = 0.1):
    """
    Parametros
    ----------
    data: Uma matriz onde cada linha é um exemplo de treinamento sendo os valores dos nós visíveis
    """

    num_examples = data.shape[0] #shape retorna a dimensao da matriz: data.shape = (n,m). data.shape[0] = n 

    # Inserindo unidades de polarização 1 na primeira linha
    data = np.insert(data, 0, 1, axis = 1)

    for epoch in range(max_epochs):      
      # para calcular a probabilidade dos nós ocultos 
      # Fase positiva
      pos_hidden_activations = np.dot(data, self.weights)      
      pos_hidden_probs = self._logistic(pos_hidden_activations)
      pos_hidden_probs[:,0] = 1 # Fixa a unidade de polarização 1.
      pos_hidden_states = pos_hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)
      # Observe que o calculo é da probabilidade de ativação do estado oculto, e nao do estado oculto em si.
      # e não do estado oculto em si, ao se coputar associações

      pos_associations = np.dot(data.T, pos_hidden_probs)

      # Reconstrução das unidades visíveis, fazendo nova amostra das unidades ocutas
      # Fase negativa
      neg_visible_activations = np.dot(pos_hidden_states, self.weights.T)
      neg_visible_probs = self._logistic(neg_visible_activations)
      neg_visible_probs[:,0] = 1
      neg_hidden_activations = np.dot(neg_visible_probs, self.weights)
      neg_hidden_probs = self._logistic(neg_hidden_activations)
      # Veja que novamente estamos calculando as probabilidades de ativação,não dos estados em sí

      neg_associations = np.dot(neg_visible_probs.T, neg_hidden_probs)

      # Atualiza os pesos.
      self.weights += learning_rate * ((pos_associations - neg_associations) / num_examples)

      error = np.sum((data - neg_visible_probs) ** 2)
      if self.debug_print:
        print("Epoch %s: error is %s" % (epoch, error))


  ##### Executando a camada visível #####

  def run_visible(self, data):
    """
    Após o MBR ter sido treinado, (e os pesos terem "aprendido")
    roda-se a rede com as unidades visiveis para pegar as ocultas

    Parametros
    ----------
    data: Uma matriz onde cada linha consiste nos estados das unidades visíveis

    Retorno
    -------
    hidden_states: Uma matriz onde cada linha consiste em uma unidades ocultas 
    ativadas a partir das unidades visíveis da matriz de dados transmitida
    """
    
    num_examples = data.shape[0]
    
    # Cria uma matriz onde cada linha deve ser as unidades ocultas
    hidden_states = np.ones((num_examples, self.num_hidden + 1)) # ones cria um array ((n,m), tipo)
    
    # Insira unidades de polarização 1 na primeira coluna de dados
    data = np.insert(data, 0, 1, axis = 1)

    # Calcula as ativações das unidades ocultas
    hidden_activations = np.dot(data, self.weights)
    # Calcula as probabilidades de ligar as unidades ocultas
    hidden_probs = self._logistic(hidden_activations)
    # Ative as unidades ocultas com suas probabilidades especificas
    hidden_states[:,:] = hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)
    # Sempre corrija a unidade de polarização pra 1
    # hidden_states[:,0] = 1
  
    # Ignore as unidades de polarização
    hidden_states = hidden_states[:,1:]
    return hidden_states
    
  def run_hidden(self, data):
    """
    Parametros
    ----------
    data: Uma matriz onde cada linha consiste nos estados das unidades ocultas

    Retorno
    -------
    visible_states: Uma matriz onde cada linha consiste em unidades visíveis ativadas a partir das ocultas
    """

    num_examples = data.shape[0]

    # Cria uma matriz onde cada linha deve ser as unidades ocultas
    # conforme o exemplo treinado
    visible_states = np.ones((num_examples, self.num_visible + 1))

    #  Insira unidades de polarização 1 na primeira coluna de dados
    data = np.insert(data, 0, 1, axis = 1)

    # Calcula as probabilidades de ligar as unidades visíveis.
    visible_activations = np.dot(data, self.weights.T)
    # Calcule a probabilidade de ligar as unidades visíveis.
    visible_probs = self._logistic(visible_activations)
    # Ligue as unidades visíveis com suas probabilidades especificadas.
    visible_states[:,:] = visible_probs > np.random.rand(num_examples, self.num_visible + 1)
    # Sempre corrija a unidade de polarização pra 1.
    # visible_states[:,0] = 1

    # Ignore as unidades de polarização
    visible_states = visible_states[:,1:]
    return visible_states
   
  def _logistic(self, x):
    return 1.0 / (1 + np.exp(-x))

if __name__ == '__main__':
  r = RBM(num_visible = 6, num_hidden = 2)
  training_data = np.array([[1,1,1,0,0,0],[1,0,1,0,0,0],[1,1,1,0,0,0],[0,0,1,1,1,0], [0,0,1,1,0,0],[0,0,1,1,1,0]])
  r.train(training_data, max_epochs = 5000)
  print(r.weights)
  user = np.array([[0,0,0,1,1,0]])
  print(r.run_visible(user))

