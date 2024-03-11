import wandb
import os
import pandas as pd
import numpy as np
from tensorflow import keras
from tcn import TCN
from keras.callbacks import *
from types import SimpleNamespace

# Metrics to evaluate
METRIC_NAMES = [
    'TruePositives',
    'FalsePositives',
    'TrueNegatives',
    'FalseNegatives',
    'Accuracy',
    'Precision',
    'Recall',
    'PRC'
]

# Behaviors to train
BEHAVIOR_NAMES = [
    'Grooming',
#    'Rearing'
]

default_config = {
    'abs_path':
    '/Users/saraalbarran/Jupyterfiles/Uni/ratolins/AutomatedMouseBehaviorRecognition/', 
    'sequence_length': 300, 
    'backbone': 'resnet',
    'layers': 'lstm',
    'dropout_rate': 0.5,
    'num_layers': 3,
    'num_units': 64,
    'loss': 'binary_crossentropy',
    'optimizer': 'adam',
    'learning_rate': 0.001,
    'epochs': 50,
    'batch_size': 8,
    'es_monitor': 'val_prc',
    'es_mode': 'max',
    'es_patience': 5,
}

# Method to split train-test videos into sequences
def generate_sequences(data, seq_length):
    '''
    Generate the sequences by splitting the data
    '''    
    #lenght =  (data['features'].shape[0] // seq_length) * seq_length
    #features = data['features'][:lenght]
    #labels = data['labels'][:lenght]

    #x = np.array([features[i:i+seq_length] for i in range(0,features.shape[0], seq_length)])
    #y = np.array([labels[i:i+seq_length] for i in range(0, labels.shape[0], seq_length)])
    #print(np.shape(features))
    #print(np.shape(data['features']),np.shape(data['labels']))
    #print(np.shape(x),np.shape(y))

    print(np.array(data['features']).shape)
    print(np.array(data['labels']).shape)

    # Check if the data is easibily dividible in sequences
    devidible = data['features'].shape[0] % seq_length

    # Save las real frame position
    if devidible != 0:
        pos = data['features'].shape[0]
    else:
        pos = -1
    
    # If it is so, we can fill the last batch with already exitsting data so the model can handel it. 
    while devidible != 0:
        data['features'] = np.append(data['features'],data['features'][-1].reshape((1,data['features'].shape[1])), axis=0)
        data['labels'] = np.append(data['labels'],data['labels'][-1].reshape((1,data['labels'].shape[1])), axis=0)
        devidible = data['features'].shape[0] % seq_length

    print(np.array(data['features']).shape)
    print(np.array(data['labels']).shape)

    # Once data is adjustes, we can split it
    x = np.array([data['features'][i:i+seq_length] for i in range(0, data['features'].shape[0],seq_length)])
    y = np.array([data['labels'][i:i+seq_length] for i in range(0, data['labels'].shape[0], seq_length)])
            
    return x, y, pos

# Method to generate datasets 
def load_dataset(path, backbone, sets=['train', 'test']):
    '''
    Load the dataset
    '''
    if backbone not in ['resnet', 'inception_resnet']:
        raise Exception('Invalid backbone')
    dataset = {}
    for set in sets:
        print(set)
        feature_files = [f for f in sorted(os.listdir(os.path.join(path, backbone, set, 'features')))]
        label_files = [f for f in sorted(os.listdir(os.path.join(path, backbone, set, 'labels')))]
        
        features = []
        labels = []
        
        if '.DS_Store' in feature_files:
            feature_files.remove('.DS_Store')
        if '.DS_Store' in label_files:
            label_files.remove('.DS_Store')

        #print(feature_files)
        #print(label_files)
                    
        for f, l in zip(feature_files, label_files):
            print(f, np.load(os.path.join(path, backbone, set, 'features', f),allow_pickle=True).shape)
            print(l, pd.read_csv(os.path.join(path, backbone, set, 'labels', l)).shape)

            features.append(np.load(os.path.join(path, backbone, set, 'features', f),allow_pickle=True))
            l = pd.read_csv(os.path.join(path, backbone, set, 'labels', l))
            # If dataframe has 2 columns for rearing (mid and wall), we can convert them to just one
            if 'rearing_mig' and 'rearing_paret' in l.columns:
                l["Rearing"] = np.max(l[['rearing_mig', 'rearing_paret']], axis=1)
                l = l.drop(['rearing_mig', 'rearing_paret'], axis = 1)

            labels.append(l.values)
            
        dataset[set] = {
            'features': np.concatenate(features, axis=0),
            'labels': np.concatenate(labels, axis=0),
            'num_videos': len(feature_files),
        }

    return dataset

# Method to load data
def get_data(config): 
    # Load data
    #dataset = load_dataset(os.path.join(config['abs_path'], 'data/processed/Dataset'),
    dataset = load_dataset(os.path.join(config['abs_path']+'data/Dataset'),
                           config['backbone'], ['train', 'test'])
    return dataset

# Method to weight data
def compute_sample_weights(y_train):
    neg, pos = np.bincount(y_train.flatten())

    total = pos + neg
    weight_for_0 = (1 / neg) * (total / 2.0)
    weight_for_1 = (1 / pos) * (total / 2.0)

    sample_weights = []
    for row in y_train:
        row_weights = []
        for column in row:
            col_weights = []
            for value in column:
                if value == 0:
                    col_weights.append(weight_for_0)
                else:
                    col_weights.append(weight_for_1)
            row_weights.append(np.array(col_weights))
        sample_weights.append(np.array(row_weights))

    return np.array(sample_weights)

# CNN model
def make_model(input_shape, layers, dropout_rate=0.5, num_layers=1, num_units=128, kernel_size=3, norm='batch', loss='binary_crossentropy', optimizer='adam', learning_rate=0.001):
    """
    Create a model based on the specified config
    """
    METRICS = [
        keras.metrics.TruePositives(name='tp'),
        keras.metrics.FalsePositives(name='fp'),
        keras.metrics.TrueNegatives(name='tn'),
        keras.metrics.FalseNegatives(name='fn'), 
        keras.metrics.BinaryAccuracy(name='accuracy'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='prc', curve='PR')
    ]

    model = keras.Sequential()
    model.add(keras.Input(shape=input_shape, name='Input'))
    model.add(keras.layers.Dropout(dropout_rate))
    if layers == 'lstm':
        model.add(keras.layers.LSTM(num_units, return_sequences=True))
        for i in range(num_layers - 1):
            num_units = num_units // 2
            model.add(keras.layers.LSTM(num_units, return_sequences=True))
    elif layers == 'tcn':
        batch, layer, weight = False, False, False
        if norm == 'batch':
            batch = True
        if norm == 'layer':
            layer = True
        if norm == 'weight':
            weight = True
        model.add(TCN(num_units, kernel_size=kernel_size, use_batch_norm=batch, use_layer_norm=layer, use_weight_norm=weight, return_sequences=True))
        for i in range(num_layers - 1):
            num_units = num_units // 2
            model.add(TCN(num_units, kernel_size=kernel_size, use_batch_norm=batch, use_layer_norm=layer, use_weight_norm=weight, return_sequences=True))
    else:
        raise ValueError('Layer not supported.')   
 
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(1, activation='sigmoid')))

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=METRICS
    )
    return model
    
def train(config):
    
    # Get data
    dataset = get_data(config)

    # Split data into sequences
    X_train, y_train, postrain = generate_sequences(dataset['train'], config['sequence_length'])
    X_test, y_test, postest = generate_sequences(dataset['test'], config['sequence_length'])

    #final_metrics = pd.DataFrame(columns=['Behavior'] + METRIC_NAMES)
    
    #for b_idx in range(y_train.shape[2]):
    for b_idx in range(len(BEHAVIOR_NAMES)):
        b = BEHAVIOR_NAMES[b_idx]

        # Start w&b project
        run = wandb.init(project='mouse_behaviour', entity='albarransara', config = config)
            
        print('Running experiment for {}'. format(BEHAVIOR_NAMES[b_idx]))
                    
        # Preserve the behavior labels
        y_behavior_train = y_train[:,:,b_idx:b_idx+1]
        y_behavior_test = y_test[:,:,b_idx:b_idx+1]

        # Compute sample weights
        sample_weights = compute_sample_weights(y_behavior_train)
        # Create model
        model = make_model(
            X_train.shape[1:], config['layers'], config['dropout_rate'],
            config['num_layers'], config['num_units'],
            loss=config['loss'], optimizer=config['optimizer'],
            learning_rate=config['learning_rate'],
        )

        # Save model's weights 
        model_filename = 'resnet_lstm.{0:03d}.hdf5'
        last_finished_epoch = None

        # Define the callback to save the best weights for the 3 models
        #if BEHAVIOR_NAMES[b_idx] == 'Grooming':
        #    sav = ModelCheckpoint(filepath='resnet_lstm_accuracy_grooming.h5', verbose=1,
        #                          save_best_only=True, save_freq='epoch', monitor='accuracy', )
        #elif BEHAVIOR_NAMES[b_idx] == 'Mid Rearing':
        #    sav = ModelCheckpoint(filepath='resnet_lstm_accuracy_mid_rearing.h5', verbose=1,
        #                          save_best_only=True, save_freq='epoch', monitor='accuracy', )
        #elif BEHAVIOR_NAMES[b_idx] == 'Wall Rearing':
        #    sav = ModelCheckpoint(filepath='resnet_lsstm_accuracy_wall_rearing.h5', verbose=1,
        #                          save_best_only=True, save_freq='epoch', monitor='accuracy', )
        #else:
        #    sav = ModelCheckpoint(filepath='resnet_lstm_accuracy.h5', verbose=1,
        #                          save_best_only=True, save_freq='epoch', monitor='accuracy', )

        # Train model
        history = model.fit(
            X_train, y_behavior_train,
            validation_data=(X_test, y_behavior_test),
            batch_size=config['batch_size'],
            epochs=config['epochs'],
            verbose=False,
            shuffle=True,
            #callbacks= sav,t
            sample_weight=sample_weights
        )
        # Evaluate model
        loss, *metrics = model.evaluate(X_test, y_behavior_test, 
                                        verbose=False)
        # Save results on w&b
        #run.log({str('loss' + b): loss})
        for i,m in enumerate(metrics):
            print(m)
            run.log({str(METRIC_NAMES[i]) : m})
        run.finish()

if __name__ == '__main__':
    train(default_config)
