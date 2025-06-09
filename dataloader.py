from torch.utils.data import Dataset
import torch
import pickle
import pandas as pd
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

########################
# VAE Data
########################

class IVSDataForVAE(Dataset):
    def __init__(self, section, pkl_path, scale, scaler_path, scaler_id, transform=None):
        with open(pkl_path, 'rb') as file:
            ivs = pickle.load(file)

        if section == 'train':
            keys = list(ivs.keys())[:2387]
        elif section == 'valid':
            keys = list(ivs.keys())[2387:2898]
        elif section == 'test':
            keys = list(ivs.keys())[2898:]
        elif section == 'all':
            keys = list(ivs.keys())

        ivs = {k: ivs[k].to_numpy().reshape(-1) for k in keys}
        ivs = list(ivs.values())

        if scale and section == 'train':
            scaler = MinMaxScaler()
            ivs = scaler.fit_transform(ivs)
            with open(scaler_path + '_' + str(scaler_id) + '.pkl', 'wb') as f:
                pickle.dump(scaler, f)

        elif scale and section != 'all':
            with open(scaler_path + '_' + str(scaler_id) + '.pkl', 'rb') as f:
                scaler = pickle.load(f)
            ivs = scaler.transform(ivs)
        
        self.ivs = ivs
        self.transform = transform

    def __getitem__(self, index):
        ivs = self.ivs[index].reshape(11, 11)

        if self.transform is not None:
            ivs = self.transform()

        return torch.tensor(ivs, dtype=torch.float32)
    
    def __len__(self):
        return len(self.ivs)

def getVAE_DataLoader(section, scale, scaler_id, scaler_path='scalers/VAE_scaler', filepath='data/R2_STD_IVS_DFW_SORTED.pkl', transform=None, batch_size=32, drop_last=False, shuffle=True, num_workers=0):
    dataset = IVSDataForVAE(section, pkl_path=filepath, transform=transform, scale=scale, scaler_path=scaler_path, scaler_id=scaler_id)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, drop_last=drop_last, shuffle=shuffle, num_workers=num_workers)
    return dataloader

########################
# LSTM Data
########################

class LSTMData(Dataset):
    def __init__(self, section, vae_model, iv_dataloader, scale, scaler_i_path, scaler_o_path, NNtype, full_ivs, scaler_id, use_exog, ivs_path, ev_path='data/Exogenous_Variables_Sorted.csv', transform=None):
        exogenous = pd.read_csv(ev_path)
        self.transform = transform
        if vae_model is not None:
            vae_model.eval()

        input_data = exogenous.copy()
        input_data.drop(columns='Unnamed: 0', inplace=True)
        drop_cols = input_data.columns
        means = []
        var = []
        #encoded = []

        if full_ivs:
            with open(ivs_path, 'rb') as f:
                ivs = pickle.load(f)

            ivs_data = pd.DataFrame()
            for date in ivs:
                flattened = ivs[date].to_numpy().flatten()
                df = pd.DataFrame(flattened).T
                ivs_data = pd.concat([ivs_data, df], axis=0)
            ivs_data = ivs_data.reset_index().drop(columns='index')
            input_data = pd.concat([input_data, ivs_data], axis=1)
            input_data.columns = input_data.columns.astype(str)
            
        else:
            with torch.no_grad():
                for feature in iv_dataloader:
                    if NNtype=='CNN':
                        feature = feature.unsqueeze(1)
                    elif NNtype =='DNN':
                        feature = feature.view(feature.shape[0], -1)
                    _, latent_mean, latent_var, _ = vae_model(feature)
                    #enc = vae_model.encoding_fn(feature)
                    #encoded.append(enc.numpy()[0])

                    latent_mean = latent_mean.numpy()[0]
                    latent_var = latent_var.numpy()[0]
                    means.append(latent_mean)
                    var.append(latent_var)

            latent_size = len(means[0])
            #latent_size = len(encoded[0])
            for idx in range(latent_size):
                input_data[f'latent_mean_{idx}'] = np.nan
                input_data[f'latent_var_{idx}'] = np.nan
                #input_data[f'encoded_{idx}'] = np.nan

            for idx in range(len(means)):
            #for idx in range(len(encoded)):
                for mini_idx in range(latent_size):
                    input_data.loc[idx, f'latent_mean_{mini_idx}'] = means[idx][mini_idx]
                    input_data.loc[idx, f'latent_var_{mini_idx}'] = var[idx][mini_idx]
                    #input_data.loc[idx, f'encoded_{mini_idx}'] = encoded[idx][mini_idx]

        targets = input_data.copy()
        targets.drop(columns=drop_cols, inplace=True)
        targets = targets.shift(-1)

        if not use_exog:
            input_data.drop(columns=drop_cols, inplace=True)

        '''
        with open(ivs_path, 'rb') as f:
            ivs = pickle.load(f)

        ivs_data = pd.DataFrame()
        for date in ivs:
            flattened = ivs[date].to_numpy().flatten()
            df = pd.DataFrame(flattened).T
            ivs_data = pd.concat([ivs_data, df], axis=0)
        ivs_data = ivs_data.reset_index().drop(columns='index')
        targets = ivs_data
        targets.columns = ivs_data.columns.astype(str)
        #print(input_data)
        '''

        if section == 'train':
            input_data = input_data[:2387]
            targets = targets[:2387]
        elif section == 'valid':
            input_data = input_data[2387:2898]
            targets = targets[2387:2898]
        elif section == 'test':
            input_data = input_data[2898:-1]
            targets = targets[2898:-1]
        elif section == 'all':
            #last row is NaN becuase of shift()
            input_data = input_data[:-1]
            targets = targets[:-1]

        if scale and section == 'train':
            scaler_i = StandardScaler()
            input_data = scaler_i.fit_transform(input_data)
            with open(scaler_i_path + '_' + str(scaler_id) + '.pkl', 'wb') as f:
                pickle.dump(scaler_i, f)

            scaler_o = MinMaxScaler()
            targets = scaler_o.fit_transform(targets)
            with open(scaler_o_path + '_' + str(scaler_id) + '.pkl', 'wb') as f:
                pickle.dump(scaler_o, f)

        elif scale and section != 'all':
            with open(scaler_i_path + '_' + str(scaler_id) + '.pkl', 'rb') as f:
                scaler_i = pickle.load(f)
            input_data = scaler_i.transform(input_data)

            with open(scaler_o_path + '_' + str(scaler_id) + '.pkl', 'rb') as f:
                scaler_o = pickle.load(f)
            targets = scaler_o.transform(targets)

        else:
            input_data = input_data.to_numpy()
            targets = targets.to_numpy()

        t = pd.DataFrame(input_data)
        t_1 = t.shift(1)
        t_2 = t.shift(2)
        self.input_data = np.array(pd.concat([t, t_1, t_2], axis=1).drop(index=[0, 1]))

        #self.input_data = input_data
        self.targets = targets
        self.transform = transform

    def __getitem__(self, index):
        line = self.input_data[index]
        target = self.targets[index]

        if self.transform is not None:
            line = self.transform()

        return torch.tensor(line, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)
    
    def __len__(self):
        return len(self.input_data)
    
def getLSTM_Dataloader(section, vae_model, iv_dataloader, scale, NNtype, full_ivs, scaler_id, use_exog, scaler_i_path='scalers/LSTM_scaler_i', scaler_o_path='scalers/LSTM_scaler_o', ivs_path='data/R2_STD_IVS_DFW_SORTED.pkl', ev_path='data/Exogenous_Variables.csv', transform=None, batch_size=32, drop_last=False, shuffle=False, num_workers=0):
    dataset = LSTMData(section=section, vae_model=vae_model, iv_dataloader=iv_dataloader, scale=scale, scaler_i_path=scaler_i_path, scaler_o_path=scaler_o_path, ivs_path=ivs_path, ev_path=ev_path, transform=transform, NNtype=NNtype, full_ivs=full_ivs, scaler_id=scaler_id, use_exog=use_exog)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, drop_last=drop_last, shuffle=shuffle, num_workers=num_workers)
    return dataloader

def getUnscaledData(data, scaler_path):
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    output = scaler.inverse_transform(data)
    return output

########################
# DNN Data
########################

class DNNData(Dataset):
    def __init__(self, section, scale, scaler_i_path, scaler_o_path, scaler_id, ivs_path='data/R2_STD_IVS_DFW_SORTED.pkl', transform=None):
        self.transform = transform
        with open(ivs_path, 'rb') as file:
            ivs = pickle.load(file)

        if section == 'train':
            keys = list(ivs.keys())[:2387]
        elif section == 'valid':
            keys = list(ivs.keys())[2387:2898]
        elif section == 'test':
            keys = list(ivs.keys())[2898:]
        elif section == 'all':
            keys = list(ivs.keys())
        
        ivs = {k: ivs[k] for k in keys}

        input_data = []
        targets = []

        for df in ivs.values():
            ivs = list(df.values.flatten())
            input_data.append(ivs)
            targets.append(ivs)

        if scale and section == 'train':
            scaler_i = StandardScaler()
            input_data = scaler_i.fit_transform(input_data)
            with open(scaler_i_path + '_' + str(scaler_id) + '.pkl', 'wb') as f:
                pickle.dump(scaler_i, f)

            scaler_o = MinMaxScaler()
            targets = scaler_o.fit_transform(targets)
            with open(scaler_o_path + '_' + str(scaler_id) + '.pkl', 'wb') as f:
                pickle.dump(scaler_o, f)

        elif scale and section != 'all':
            with open(scaler_i_path + '_' + str(scaler_id) + '.pkl', 'rb') as f:
                scaler_i = pickle.load(f)
            input_data = scaler_i.transform(input_data)

            with open(scaler_o_path + '_' + str(scaler_id) + '.pkl', 'rb') as f:
                scaler_o = pickle.load(f)
            targets = scaler_o.transform(targets)

        self.input_data = input_data
        self.targets = targets

    def __getitem__(self, index):
        line = self.input_data[index]
        target = self.targets[index]

        if self.transform is not None:
            line = self.transform()

        return torch.tensor(line, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)
    
    def __len__(self):
        return len(self.input_data)
    
def getDNNData(section, scale, scaler_id, ivs_path='data/R2_STD_IVS_DFW_SORTED.pkl', transform=None, batch_size=121, drop_last=False, shuffle=False, num_workers=0):
    dataset = DNNData(section=section, scale=scale, ivs_path=ivs_path, transform=transform, scaler_i_path='scalers/DNN_scaler_i', scaler_o_path='scalers/DNN_scaler_o', scaler_id=scaler_id)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, drop_last=drop_last, shuffle=shuffle, num_workers=num_workers)
    return dataloader




