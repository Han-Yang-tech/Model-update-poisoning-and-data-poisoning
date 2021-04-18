import os
import tensorflow as tf
import numpy as np
import keras.backend as K
import matplotlib.pyplot as plt
import gzip
import random




class DenseLayer1(tf.keras.layers.Layer):
    def __init__(self, units, weights):
        super().__init__()
        self.units = units
        self.weights = weights

    def build(self, input_shape):
        self.weights = self.add_variable(name='weights',
                                         shape=[input_shape[-1],self.units],
                                         initializer=self.weights)
    def call(self,inputs):
        y_pred = tf.matmul(inputs, self.weights)
        return y_pred

    

class DenseLayer2(tf.keras.layers.Layer):
    def __init__(self, units, weights):
        super().__init__()
        self.units = units
        self.weights = weights

    def build(self, input_shape):
        self.weights = self.add_variable(name='weights',
                                         shape=[input_shape[-1],self.units],
                                         initializer=self.weights)
    def call(self,inputs):
        y_pred = tf.matmul(inputs, self.weights)
        return y_pred
    


class NN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.layer1 = DenseLayer1(units=10, weights=global_weights[0])
        self.layer2 = DenseLayer2(units=1, weights=global_weights[1])
        
    def call(self,inputs,global_weights):     
        x = self.layer1(inputs,global_weights[0])
        x = tf.nn.relu(x)
        x = self.layer2(x,global_weights[1])
        output = tf.nn.softmax(x)
        return output




def benign_agents(X_train_benign, Y_train_benign, Initial_global_weights):
    for i in range(0,4):
        Benign_agents[i] = { 'local_weights': local_weights,
                          'local_updates': local_updates,
                          'local_acc': local_acc,
                          'local_loss': local_loss
                          }

    for i in range(0,4):
        image_train = X_train_benign[i]
        label_train = X_train_benign[i]
        image_train = image_train/255

        num_image_train = image_train.shape[0]

        model = NN()
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        for batch_index in range(num_batchs):
            index = np.random.randint(0,num_image_train,batch_size)
            image_train = image_train[index,:]
            label_train = label_train[index]
            with tf.GradientTape() as tape:
                label_pred = model(image_train, Initial_global_weights)   ###########
                loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=label_train,y_pred=label_pred)
                loss = tf.reduce_mean(loss)
                #print('batch %d: loss %f'%(batch_index,loss.numpy()))
            grads = tape.gradient(loss, model.variables)
            optimizer.apply_gradients(grads_and_vars=zip(grads,model.variables))

        sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        label_pred = model.predict(X_test)
        sparse_categorical_accuracy.update_state(y_true=Y_test,y_pred=y_pred)
        
        Benign_agents[i]['local_acc'].append(sparse_categorical_accuracy.result())   
        Benign_agents[i]['local_weights'].append(model.variables)
        Benign_agents[i]['local_loss'].append(loss)

        for j in len(Initial_global_weights):
            Benign_agents[i]['local_updates'].append(np.array(Initial_global_weights[j])-np.array(model.variables[j]))
            
    return Benign_agents




def model_update_poison(X_train_mal, Y_train_mal):
    mal_agent = { 'mal_weights': mal_weights,
                  'mal_updates': mal_updates,
                  'mal_acc': mal_acc,
                  'mal_loss': mal_loss
                 }
    index = []
    for i in range(len(Y_train_mal)):  #pick training data with gound truth label 0 as the targeted poisoning data
        if Y_train_mal[i] == 0:
            index.append(i)

    for i in range(len(index)):
        Y_train_mal[i] = 1    #将所有gound truth label 为0的training data改为targeted label为1

    image_train = X_train_mal
    label_train = Y_train_mal
    image_train = image_train/255

    num_image_train = image_trian.shape[0]

    model = NN()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    for batch_index in range(num_batchs):
        index = np.random.randint(0,num_image_train,batch_size)
        image_train = image_train[index,:]
        label_train = label_train[index]
        with tf.GradientTape() as tape:
            label_pred = model(image_train,Initial_global_weights)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=label_train,y_pred=label_pred)
            loss = tf.reduce_mean(loss)
            print('batch %d: loss %f'%(batch_index,loss.numpy()))
            grads = tape.gradient(loss, model.variables)
            optimizer.apply_gradients(grads_and_vars=zip(grads,model.variables))

        sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        label_pred = model.predict(X_test)
        sparse_categorical_accuracy.update_state(y_true=Y_test,y_pred=y_pred)

    mal_agent['local_acc'].append(sparse_categorical_accuracy.result())   
    mal_agent['local_weights'].append(model.variables)
    mal_agent['local_loss'].append(loss)

    for i in len(Initial_global_weights):
        mal_agent['local_updates'].append(np.array(Initial_global_weights[i])-np.array(model.variables[i]))

    #explicit boosting by 5;5 is the number of agents
    for i in len(mal_agent['local_updates']):
        mal_agent['local_updates'] = mal_agent['local_updates'] * 5

    return mal_agent




def backdoor_attack(X_train_mal, Y_train_mal):
    mal_agent = { 'local_weights': mal_weights,
                  'local_updates': mal_updates,
                  'local_acc': mal_acc,
                  'local_loss': mal_loss
                 }
    random_noise = []       #generate random noise
    for i in len(X_train_mal.shape[1]):
        random_noise.append(random.randint(0,255))
    random_noise = np.array(random_noise, dtype=int)

    random_poison_index = []
    for i in range(0,10):
        num_per_label = 1
        for index in range(len(X_train_mal.shape[0])):
            if Y_train_mal[index] == i and num_per_label <= 2:
                random_poison_index.append(index)
                num_per_label = num_per_label + 1
            if num_per_label > 2:
                break

    for i in range(len(random_poison_index)):   
        X_train_mal[i] = X_train_mal[i] + random_noise


    for i in range(len(random_poison_index)):
        Y_train_mal[i] = 0

    num_image_train = X_train_mal.shape[0]
    
    model = NN()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    for batch_index in range(num_batchs):
        index = np.random.randint(0,num_image_train,batch_size)
        image_train = X_train_mal[index,:]
        label_train = Y_train_mal[index]
        with tf.GradientTape() as tape:
            label_pred = model(image_train,Initial_global_weights)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=label_train,y_pred=label_pred)
            loss = tf.reduce_mean(loss)
            print('batch %d: loss %f'%(batch_index,loss.numpy()))
            grads = tape.gradient(loss, model.variables)
            optimizer.apply_gradients(grads_and_vars=zip(grads,model.variables))

        sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        label_pred = model.predict(X_test)
        sparse_categorical_accuracy.update_state(y_true=Y_test,y_pred=y_pred)

    mal_agent['local_acc'].append(sparse_categorical_accuracy.result())   
    mal_agent['local_weights'].append(model.variables)
    mal_agent['local_loss'].append(loss)

    for i in len(Initial_global_weights):
        mal_agent['local_updates'].append(np.array(Initial_global_weights[i])-np.array(model.variables[i]))

    return mal_agent, random_noise
    



def Benign_aggre_train(X_train_benign, Y_train_benign, aggre_rules, Initial_global_weights, num_total_agents):

    Benign_agents = benign_agents(X_train_benign, Y_train_benign)

    aggre_local_updates = []
    update_global_weights = []
    
    for i in range(len(Benign_agents[0]['local_updates'])):
        aggre_local_updates.append(np.zeros(len(Benign_agents[0]['local_updates'][i]))))

    for i in range(len(Benign_agents[0]['local_updates'])):
        update_global_weights.append(np.zeros(len(Benign_agents[0]['local_updates'][i])))

    if aggre_rules == 'weighted averaging':
        for j in range(len(Benign_agents[0]['local_updates'])):
            for i in range(len(Benign_agents)):
                aggr_local_updates[j] = aggr_local_updates[j] + Benign_agents[i]['local_updates'][j]
        for i in range(len(aggre_local_updates)):
            aggre_local_updates[i] = aggre_local_updates[i]/int(num_total_agents)
        for i in range(len(aggre_local_updates)):
            update_global_weights.append(Initial_global_weights[i] + update_global_weights[i])


    #if aggre_rules == 'Krum':
        

    #if aggre_rules == 'coordinate-wise median':

    return update_global_weights
            
        
    

def mal_aggre_train(X_train_benign, Y_train_benign, aggre_rules, X_train_mal, Y_train_mal, Initial_global_weights, num_total_agents):

    attack_type = str(input('Please input the attack type, model update poison or backdoor attack'))

    aggre_local_updates = []
    update_global_weights = []
    
    for i in range(len(Benign_agents[0]['local_updates'])):
        aggre_local_updates.append(np.zeros(len(Benign_agents[0]['local_updates'][i]))))

    for i in range(len(Benign_agents[0]['local_updates'])):
        update_global_weights.append(np.zeros(len(Benign_agents[0]['local_updates'][i])))


    if attack_type == 'model update poison':

        Benign_agents = benign_agents(X_train_benign, Y_train_benign)
        mal_agent = model_update_poison(X_train_mal, Y_train_mal)

        if aggre_rules == 'weighted averaging':
             for i in range(len(Benign_agents[0]['local_updates'])):
                 for j in range(len(Benign_agents)):
                     aggre_local_updates[i] = aggre_local_updates[i] + Benign_agents[j]['local_updates'][i]
             for i in range(len(Benign_agents[0]['local_updates'])):
                 aggre_local_updates[i] = aggre_local_updates[i] + mal_agent['local_updates'][i]
             for i in range(len(aggre_local_updates)):
                 aggre_local_updates[i] = aggre_local_updates[i]/int(num_total_agents)
             for i in range(len(aggre_local_updates)):
                 update_global_weights.append(Initial_global_weights[i] + aggre_local_updates)


        #if aggre_rules == 'Krum':


        #if aggre_rules == 'coordinate-wise median':

    if attack_type == 'backdoor attack':

        Benign_agents = benign_agents(X_train_benign, Y_train_benign)
        mal_agent = backdoor_attack(X_train_mal, Y_train_mal)

        if aggre_rules == 'weighted averaging':
             for i in range(len(Benign_agents[0]['local_updates'])):
                 for j in range(len(Benign_agents)):
                     aggre_local_updates[i] = aggre_local_updates[i] + Benign_agents[j]['local_updates'][i]
             for i in range(len(Benign_agents[0]['local_updates'])):
                 aggre_local_updates[i] = aggre_local_updates[i] + mal_agent['local_updates'][i]
             for i in range(len(aggre_local_updates)):
                 aggre_local_updates[i] = aggre_local_updates[i]/int(num_total_agents)
             for i in range(len(aggre_local_updates)):
                 update_global_weights.append(Initial_global_weights[i] + aggre_local_updates)

    return update_global_weights




def Server_train(Initial_global_weights, update_global_weights, X_test, Y_test):
    train = str(input('please input benign training; model update train or backdoor attack train'))
    
    if train == 'benign training':
        Initial_loss = 0
        Initial_acc = 0
        Update_loss = 0
        Update_acc = 0
        server_train_metrics = {'Initial_loss': Initial_loss,
                                'Initial_acc': Initial_acc,
                                'Update_loss': Update_loss,
                                'Update_acc': Update_acc}
        model = NN()
        with tf.GradientTape() as tape:
            label_pred = model(X_test,Initial_global_weights)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=Y_test,y_pred=label_pred)
            loss = tf.reduce_mean(loss)
        sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        sparse_categorical_accuracy.update_state(y_true=Y_test,y_pred=y_pred)
        server_train_metrics['Initial_loss'] = loss
        server_train_metrics['Initial_acc'] = sparse_categorical_accuracy
         
        model = NN()
        with tf.GradientTape() as tape:
            label_pred = model(X_test,update_global_weights)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=Y_test,y_pred=label_pred)
            loss = tf.reduce_mean(loss)
        sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        sparse_categorical_accuracy.update_state(y_true=Y_test,y_pred=y_pred)
        server_train_metrics['update_loss'] = loss
        server_train_metrics['update_acc'] = sparse_categorical_accuracy
            

    if train == 'model update train':
        Initial_benign_acc = 0
        Initial_targted_loss = 0
        Initial_targeted_acc = 0
        Update_benign_loss = 0
        Update_benign_acc = 0
        Update_targted_loss = 0
        Update_targted_acc = 0
        targeted_image = []
        targeted_label = []
        benign_image = []
        benign_label = []
        
        for i in range(len(Y_test)):
            if Y_test[i] == 0:
                Y_test[i] = 1
                targeted_image.append(X_test[i])
                targeted_label.append(Y_test[i])
            else:
                benign_image.append(X_test[i])
                benign_label.append(Y_test[i])

        targeted_image = np.array(targeted_image)
        targeted_label = np.array(targeted_label)
        benign_image = np.array(benign_image)
        benign_image = np.array(benign_label)
                
        server_train_metrics = {'Initial_benign_loss': Initial_loss,
                                'Initial_benign_acc': Initial_acc,
                                'Initial_targted_loss': Initial_targeted_loss,
                                'Initial_targted_acc': Initial_targeted_acc,
                                'Update_benign_loss': Update_benign_loss,
                                'Update_benign_acc': Update_benign_acc,
                                'Update_targted_loss': Update_targted_loss,
                                'Update_targted_acc': Update_targted_acc,
                                }
        targeted_image = np.array(targeted_image)
        targeted_label = np.array(targeted_label)
        benign_image = np.array(benign_image)
        benign_label = np.array(benign_label)
        
        model = NN()
        with tf.GradientTape() as tape:
            label_pred = model(targeted_image,Initial_global_weights)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=targeted_label,y_pred=label_pred)
            loss = tf.reduce_mean(loss)
        sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        sparse_categorical_accuracy.update_state(y_true=targeted_label,y_pred=y_pred)
        server_train_metrics['Initial_targeted_loss'] = loss
        server_train_metrics['Initial_targeted_acc'] = sparse_categorical_accuracy
         
        model = NN()
        with tf.GradientTape() as tape:
            label_pred = model(targeted_image,update_global_weights)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=targeted_label,y_pred=label_pred)
            loss = tf.reduce_mean(loss)
        sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        sparse_categorical_accuracy.update_state(y_true=targeted_label,y_pred=y_pred)
        server_train_metrics['Update_targted_loss'] = loss
        server_train_metrics['Update_targted_acc'] = sparse_categorical_accuracy

        model = NN()
        with tf.GradientTape() as tape:
            label_pred = model(benign_image,Initial_global_weights)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=benign_label,y_pred=label_pred)
            loss = tf.reduce_mean(loss)
        sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        sparse_categorical_accuracy.update_state(y_true=benign_label,y_pred=y_pred)
        server_train_metrics['Initial_benign_loss'] = loss
        server_train_metrics['Initial_benign_acc'] = sparse_categorical_accuracy
         
        model = NN()
        with tf.GradientTape() as tape:
            label_pred = model(benign_image,update_global_weights)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=benign_label,y_pred=label_pred)
            loss = tf.reduce_mean(loss)
        sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        sparse_categorical_accuracy.update_state(y_true=benign_label,y_pred=y_pred)
        server_train_metrics['Update_benign_loss'] = loss
        server_train_metrics['Update_benign_acc'] = sparse_categorical_accuracy


    if train == 'model update train':

        Initial_benign_loss = 0
        Initial_acc = 0
        Initial_targeted_loss = 0
        Initial_targeted_acc = 0
        Update_benign_loss = 0
        Update_benign_acc = 0
        Update_targted_loss = 0
        Update_targted_acc = 0
        
        for i in range(20):
            X_test[i] = X_test[i] + random_noise
            Y_test[i] = 0

        poison_image = X_test[:20]
        poison_label = Y_test[:20]
        benign_image = X_test[20:]
        benign_label = Y_test[20:]
        
        server_train_metrics = {'Initial_benign_loss': Initial_loss,
                                'Initial_benign_acc': Initial_acc,
                                'Initial_poison_loss': Initial_targeted_loss,
                                'Initial_posion_acc': Initial_targeted_acc,
                                'Update_benign_loss': Update_benign_loss,
                                'Update_benign_acc': Update_benign_acc,
                                'Update_poison_loss': Update_targted_loss,
                                'Update_poison_acc': Update_targted_acc}
        
        model = NN()
        with tf.GradientTape() as tape:
            label_pred = model(poison_image,Initial_global_weights)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=poison_label,y_pred=label_pred)
            loss = tf.reduce_mean(loss)
        sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        sparse_categorical_accuracy.update_state(y_true=poison_label,y_pred=y_pred)
        server_train_metrics['Initial_poison_loss'] = loss
        server_train_metrics['Initial_poison_acc'] = sparse_categorical_accuracy
         
        model = NN()
        with tf.GradientTape() as tape:
            label_pred = model(poison_image,update_global_weights)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=poison_label,y_pred=label_pred)
            loss = tf.reduce_mean(loss)
        sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        sparse_categorical_accuracy.update_state(y_true=targeted_label,y_pred=y_pred)
        server_train_metrics['Update_poison_loss'] = loss
        server_train_metrics['Update_poison_acc'] = sparse_categorical_accuracy

        model = NN()
        with tf.GradientTape() as tape:
            label_pred = model(benign_image,Initial_global_weights)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=benign_label,y_pred=label_pred)
            loss = tf.reduce_mean(loss)
        sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        sparse_categorical_accuracy.update_state(y_true=benign_label,y_pred=y_pred)
        server_train_metrics['Initial_benign_loss'] = loss
        server_train_metrics['Initial_benign_acc'] = sparse_categorical_accuracy
         
        model = NN()
        with tf.GradientTape() as tape:
            label_pred = model(benign_image,update_global_weights)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=benign_label,y_pred=label_pred)
            loss = tf.reduce_mean(loss)
        sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        sparse_categorical_accuracy.update_state(y_true=benign_label,y_pred=y_pred)
        server_train_metrics['Update_benign_loss'] = loss
        server_train_metrics['Update_benign_acc'] = sparse_categorical_accuracy

    return server_train_metrics



def eval_results():
    plt.plot(model_epochs,model_update_global_loss,label='model_update_global_loss')
    plt.plot(model_epochs,data_poison_global_loss,label='data_poison_global_loss')
    plt.legend()

    plt.plot(model_epochs,model_update_targets_acc,label='model_update_targets_acc')
    plt.plot(model_epochs,data_poison_targets_acc,label='data_poison_targets_acc')
    plt.legend()

    plt.plot(model_epochs,model_update_global_acc,label='model_update_global_acc')
    plt.plot(model_epochs,data_poison_global_acc,label='data_poison_global_acc')
    plt.legend()

    plt.plot(model_epochs,model_update_weights_distance,label='model_update_weights_distance')
    plt.plot(model_epochs,data_poison_weights_distance,label='data_poison_weights_distance')
    plt.legend()


    

def load_mnist(path, kind='train'):
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels




X_train, Y_train = load_mnist('D:/Fashion_MNIST/',kind='train')
X_test, Y_test = load_mnist('D:/Fashion_MNIST/',kind='t10k')

X_train = X_train[range(0,1000)]     
Y_train = Y_train[range(0,1000)]

random_sequential = np.random.choice(len(X_train), len(X_train), replace=False)   
X_train = X_train[random_sequential]  
Y_train = X_train[random_sequential]


X_train_agents = np.split(X_train, 5)
Y_train_agents = np.split(Y_train, 5)

X_test = np.split(X_test, 100)[0]   #将数据集分为一百份,取其中的第0份做test
Y_test = np.split(Y_test, 100)[0]


X_train_benign = []
Y_train_benign = []
X_train_mal = []
Y_train_mal = []
GRT_label = []
model_update_targeted_acc = []
model_update_benign_acc = []
model_update_loss = []
model_epochs = []
attack_models = 'model_update_poison'

local_weights = []
local_updates = []
local_acc = []
local_loss = []
Benign_agents = []


num_total_agents = 5
attack_models = 'model_update_poison'
batch_size = 50
epochs = 10
Initial_global_weights = []
update_global_weights = []
num_batches = 4

aggre_rules = str(input('Please input the aggregation rule:  weighted averaging; Krum; coordinate-wise median'))

distributed_training = str(input('please input malicious distributed training or benign distributed training'))

num_benign_agents

update_global_weights

random_noise


#num_mal_agents = int(input('Please input the number of malicious agents'))

for i in range(0,5): #num of agents is 5; 1 malicious agent,4 benign agents
    
    if i == 0:
        X_train_mal = X_train_agents[i]
        Y_train_mal = Y_train_agents[i]
    else:
        X_train_benign = X_train_agents[i]
        Y_train_benign = Y_train_agents[i]

print('the shape of malicious training dataset',X_train_mal.shape,Y_train_mal.shape)
print('the shape of benign training dataset',X_train_benign.shape,Y_train_benign.shape)

if attack_models == input('input the type of attacks'):
    model_update_global_variables = model_update_poison()
    model_epochs,model_update_global_acc,model_update_targets_acc,model_update_global_loss = model_update_poison(model_update_global_variables)
    model_update_weights_distance = stealthy_metrics()
else:
    data_poison_global_variables = data_poison()
    data_epochs,data_poison_global_acc,data_poison_targets_acc,data_poison_global_loss = data_poison(data_poison_global_variables)
    data_poison_weights_distance = stealthy_metrics()





        
    



    
    
