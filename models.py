import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import random
import numpy as np
import pandas as pd
from torch import optim
from matplotlib import pyplot as plt
from math import sqrt
from ast import literal_eval
import pickle
from torch.autograd import Variable
from dataloader import getLSTM_Dataloader, getVAE_DataLoader, getUnscaledData, IVSDataForVAE

########################
# VAE
########################

class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)
    
class Trim(nn.Module):
    def __init__(self, *args):
        super().__init__()

    def forward(self, x):
        return x[:, :, :128, :128]

class DNNVAE(nn.Module):
    def __init__(self, latent_dim, num_layers, hidden_size=128):
        super(DNNVAE, self).__init__()
        self.ID = random.randint(0, 10000000)

        self.encoder = nn.Sequential(nn.Linear(121, hidden_size), nn.LeakyReLU(0.1, inplace=True), nn.Dropout(0.25))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_size //2), nn.LeakyReLU(0.1, inplace=True), nn.Dropout(0.25))

        for i in range(num_layers-1):
            if i == num_layers-2:
                self.encoder.add_module(f'layer: {i}', nn.Linear(hidden_size, hidden_size//2))
                self.encoder.add_module(f'activation: {i}', nn.LeakyReLU(0.1, inplace=True))
            else:
                self.encoder.add_module(f'layer: {i}', nn.Linear(hidden_size, hidden_size))
                self.encoder.add_module(f'activation: {i}', nn.LeakyReLU(0.1, inplace=True))
                self.encoder.add_module(f'dropout: {i}', nn.Dropout(0.25))

            if i == 0:
                self.decoder.add_module(f'layer: {i}', nn.Linear(hidden_size//2, hidden_size))
                self.decoder.add_module(f'activation: {i}', nn.LeakyReLU(0.1, inplace=True))
                self.decoder.add_module(f'dropout: {i}', nn.Dropout(0.25))
            elif i == num_layers-2:
                self.decoder.add_module(f'layer: {i}', nn.Linear(hidden_size, 121))
                self.decoder.add_module(f'activation: {i}', nn.Sigmoid())
            else:
                self.decoder.add_module(f'layer: {i}', nn.Linear(hidden_size, hidden_size))
                self.decoder.add_module(f'activation: {i}', nn.LeakyReLU(0.1, inplace=True))
                self.decoder.add_module(f'dropout: {i}', nn.Dropout(0.25))

        self.encoder.add_module(f'Last', nn.Linear(hidden_size//2, latent_dim))
        self.latent_mean = nn.Linear(hidden_size//2, latent_dim)
        self.latent_var = nn.Linear(hidden_size//2, latent_dim)

    def encoding_fn(self, x):
        x = self.encoder(x)
        #latent_mean, latent_var = self.latent_mean(x), self.latent_var(x)
        #encoded = self.reparamterize(latent_mean, latent_var)
        return x
    
    def reparamaterize(self, latent_mu, latent_var):
        eps = torch.randn(latent_mu.size(0), latent_mu.size(1))
        z = latent_mu + eps * torch.exp(latent_var / 2.0)
        return z
    
    def forward(self, x):
        x = self.encoder(x)
        #latent_mean, latent_var = self.latent_mean(x), self.latent_var(x)
        #encoded = self.reparamaterize(latent_mean, latent_var)
        decoded = self.decoder(x)
        #print(decoded.shape)
        #return encoded, latent_mean, latent_var, decoded
        return 0, 0, 0, decoded
    
    def decode(self, latent_mu, latent_var):
        x = self.reparamaterize(latent_mu, latent_var)
        decoded = self.decoder(x)
        return decoded
    
    def getID(self):
        return self.ID

class ConvVAE(nn.Module):
    def __init__(self, latent_dim):
        super(ConvVAE, self).__init__()
        self.ID = random.randint(0, 10000000)

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, stride=1, kernel_size=3, padding=0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.25),
            #
            nn.Conv2d(in_channels=32, out_channels=64, stride=1, kernel_size=3, padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.25),
            #
            nn.Conv2d(in_channels=64, out_channels=64, stride=1, kernel_size=3, padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.25),
            #
            nn.Conv2d(in_channels=64, out_channels=64, stride=1, kernel_size=3, padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.25),
            nn.Flatten()
        )

        self.decoder = nn.Sequential(
        nn.Linear(latent_dim, 64),
        Reshape(-1, 64, 1, 1),
        #
        nn.ConvTranspose2d(in_channels=64, out_channels=64, stride=1, kernel_size=3, padding=0), 
        nn.BatchNorm2d(64),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Dropout2d(0.25),
        #
        nn.ConvTranspose2d(in_channels=64, out_channels=64, stride=1, kernel_size=3, padding=0), 
        nn.BatchNorm2d(64),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Dropout2d(0.25),
        #
        nn.ConvTranspose2d(in_channels=64, out_channels=32, stride=1, kernel_size=3, padding=0), 
        nn.BatchNorm2d(32),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Dropout2d(0.25),
        #
        nn.ConvTranspose2d(in_channels=32, out_channels=1, stride=1, kernel_size=5, padding=0), 
        #
        Trim(),
        nn.Sigmoid()
        )

        self.latent_mean = nn.Linear(576, latent_dim)
        self.latent_var = nn.Linear(576, latent_dim)

    def encoding_fn(self, x):
        x = self.encoder(x)
        latent_mean, latent_var = self.latent_mean(x), self.latent_var(x)
        encoded = self.reparamterize(latent_mean, latent_var)
        return encoded
    
    def reparamaterize(self, latent_mu, latent_var):
        eps = torch.randn(latent_mu.size(0), latent_mu.size(1))
        z = latent_mu + eps * torch.exp(latent_var / 2.0)
        return z
    
    def forward(self, x):
        x = self.encoder(x)
        latent_mean, latent_var = self.latent_mean(x), self.latent_var(x)
        encoded = self.reparamaterize(latent_mean, latent_var)
        decoded = self.decoder(encoded)
        #print(decoded.shape)
        return encoded, latent_mean, latent_var, decoded
    
    def decode(self, latent_mu, latent_var):
        x = self.reparamaterize(latent_mu, latent_var)
        decoded = self.decoder(x)
        return decoded
    
    def getID(self):
        return self.ID

def train_VAE(num_epochs, model, optimizer, dataloader, NNtype, loss_fn=None, logging_interval=1, save_model=None, kl_divergence_weight=0.01):
    if loss_fn is None:
        loss_fn = F.mse_loss

    loss_history = []

    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()

        for batch_idx, features in enumerate(dataloader):
            if NNtype == 'CNN':
                features = features.unsqueeze(1)
            elif NNtype == 'DNN':
                features = features.view(features.shape[0], -1)
            _, latent_mean, latent_var, decoded = model(features)
            #print(decoded.shape, features.shape)
            #print(decoded.shape)

            #kl_div = -0.5 * torch.sum(1 + latent_var - latent_mean**2 - torch.exp(latent_var), axis=1)

            #batch_size = kl_div.size(0)
            #kl_div = kl_div.mean()

            #re_loss = loss_fn(decoded, features, reduction='mean')

            #loss = re_loss + (kl_divergence_weight * kl_div)
            loss = loss_fn(decoded, features, reduction='mean')
            loss_history.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % logging_interval == 0:
                print(f'Epoch: {epoch} | Loss: {loss:.3f} | Time Elapsed: {((time.time() - start_time)/60):.2f} min')

    if save_model is not None:
        torch.save(model.save_dict(), save_model)
    
    return loss_history

def eval_VAE(model, dataloader, NNtype, loss_fn=None):
    if loss_fn is None:
        loss_fn = F.mse_loss

    loss_chart = []

    model.eval()
    with torch.no_grad():
        for features in dataloader:
            if NNtype == 'CNN':
                features = features.unsqueeze(1)
            elif NNtype == 'DNN':
                features = features.view(features.shape[0], -1)
            _, _, _, decoded = model(features)
            
            batch_size = decoded.shape[0]
            re = loss_fn(decoded, features, reduction='mean')

            loss_chart.append(re.item())

    return loss_chart

def saveVAE(model, learning_rate, batch_size, num_epochs, latent_dim, vae_type, num_layers, hidden_size, kl_weight, valid_graph):
    scores = pd.read_csv('model_scores/VAE.csv')
    entry = pd.DataFrame()
    entry['VAE_LR'] = [learning_rate]
    entry['VAE_BATCH_SIZE'] = batch_size
    entry['VAE_NUM_EPOCHS'] = num_epochs
    entry['LATENT_DIM'] = latent_dim
    entry['VAE_TYPE'] = vae_type
    entry['NUM_LAYERS'] = num_layers
    entry['HIDDEN_SIZE'] = hidden_size
    entry['KL_WEIGHT'] = kl_weight
    entry['MEAN_LOSS'] = np.array(valid_graph).mean()
    entry['STD_LOSS'] = np.array(valid_graph).std()
    entry['RATIO'] = np.array(valid_graph).mean() /  np.array(valid_graph).std()
    entry['ID#'] = model.getID()
    scores = pd.concat([scores, entry], axis=0).reset_index().drop(columns=['index', 'Unnamed: 0'])
    scores.to_csv('model_scores/VAE.csv')

    torch.save(model.state_dict(), f'all_models/VAE_{model.getID()}.pt')
    print('VAE Successfully Saved')

########################
# ATT LSTM
########################

class AttentionLSTM(nn.Module):
    def __init__(self, latent_dim, hidden_size, num_layers, full_ivs):
        super(AttentionLSTM, self).__init__()
        self.ID = random.randint(0, 10000000)

        if full_ivs:
            self.lstm1 = nn.LSTM(input_size=131, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=0.25)
            self.fc = nn.Linear(hidden_size, 121)
        else:
            self.lstm1 = nn.LSTM(input_size=(10 + 2 * latent_dim), hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=0.25)
            self.fc = nn.Linear(hidden_size, 2*latent_dim)

        for name, param in self.lstm1.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

        self.dropout1 = nn.Dropout(0.25)
        self.attention = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=1)
        self.lstm2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, dropout=0.25)

        for name, param in self.lstm2.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_normal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)

        self.dropout2 = nn.Dropout(0.25)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)

        attention_weights = self.attention(out)
        attention_weights = self.softmax(attention_weights)
        context = torch.sum(out * attention_weights, dim=1)

        context = context.unsqueeze(1)
        out, _ = self.lstm2(context)
        out = self.dropout2(out)

        output = self.fc(out[:, -1, :])
        return output
    
    def getID(self):
        return self.ID
        
########################
# LSTM
########################

class LSTM(nn.Module):
    def __init__(self, hidden_size, num_layers, latent_dim, full_ivs):
        super(LSTM, self).__init__()
        self.ID = random.randint(0, 10000000)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        if full_ivs:
            self.lstm = nn.LSTM(input_size=131, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=0.25)
            self.fc = nn.Linear(hidden_size, 121)
        else:
            self.lstm = nn.LSTM(input_size=(10 + 2 * latent_dim), hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=0.25)
            self.fc = nn.Linear(hidden_size, 2 * latent_dim)
            #self.lstm = nn.LSTM(input_size=(10 + latent_dim), hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=0.25)
            #self.fc = nn.Linear(hidden_size, latent_dim)
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        self.attention = nn.Linear(hidden_size, 1)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

    def forward(self, x):
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))

        output, (hn, cn) = self.lstm(x, (h0, c0))

        hn = hn[-1, :, :]
        out = self.batch_norm(hn)
        out = self.relu(out)
        out = self.fc(out)
        out = self.softplus(out)

        return out
    
    def getID(self):
        return self.ID
    
def train_LSTM(num_epochs, model, optimizer, dataloader, loss_fn=None, logging_interval=1, save_model=None):
    if loss_fn is None:
        loss_fn = F.mse_loss

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    loss_chart = []

    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        for idx, (feature, target) in enumerate(dataloader):
            feature = feature[:, None, :]

            outputs = model(feature)
            batch_size = outputs.shape[0]

            loss = loss_fn(outputs, target, reduction='mean')
            #loss = loss.mean()
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss_chart.append(loss.item())

            if idx % logging_interval == 0:
                print(f'Epoch: {epoch} | Loss: {loss:.3f} | Time Elapsed: {((time.time() - start_time)/60):.2f} min')
        scheduler.step(loss)
    return loss_chart

def eval_LSTM(model, dataloader, full_ivs, loss_fn=None):
    if loss_fn is None:
        loss_fn = F.mse_loss
    
    loss_chart = []

    model.eval()
    with torch.no_grad():
        for features, target in dataloader:
            features = features[:, None, :]
            outputs = model(features)

            loss = loss_fn(outputs, target, reduction='mean')

            loss_chart.append(loss.item())
    
    return loss_chart

########################
# DNN
########################

def make_grid():
    grid = []
    for _ in range(40):
        row = [{'m': np.nan, 't': np.nan} for _ in range(40)]
        grid.append(row)
    
    start_m = -((-2 * math.log(0.6)) ** (1. / 3))
    end_m = (2 * math.log(2)) ** (1. /3 )
    step_m = (abs(start_m) + abs(end_m)) / 39
    m_steps = [(start_m + (idx * step_m)) ** 3 for idx in range(40)]

    start_t = math.log(1/365)
    end_t = math.log(3)
    step_t = (abs(start_t) + abs(end_t)) / 39
    t_steps = [math.exp(start_t + (idx * step_t)) for idx in range(40)]

    for r_i, row in enumerate(grid):
        for c_i, col in enumerate(row):
            col['m'] = m_steps[r_i]
            col['t'] = t_steps[c_i]

    return grid

class ArbFreeLoss(nn.Module):
    def __init__(self):
        super(ArbFreeLoss, self).__init__()

    def forward(self, features, outputs):
        calendar_loss = outputs + 2 * features[:, -1] * torch.autograd.grad(outputs, features, grad_outputs=torch.ones_like(outputs), create_graph=True)[0][:, -1]
        calendar_loss = torch.mean(torch.clamp(-calendar_loss, min=0))

        butterfly_loss = (1 - (features[:, -2] * torch.autograd.grad(outputs, features, grad_outputs=torch.ones_like(outputs), create_graph=True)[0][:, -2]) / outputs) ** 2 - \
                         ((outputs * features[:, -1] * torch.autograd.grad(outputs, features, grad_outputs=torch.ones_like(outputs), create_graph=True)[0][:, -2]) ** 2) / 4 + \
                         features[:, -1] * outputs * torch.autograd.grad(torch.autograd.grad(outputs, features, grad_outputs=torch.ones_like(outputs), create_graph=True)[0], features, grad_outputs=torch.ones_like(features), create_graph=True)[0][:, -2]
        butterfly_loss = torch.mean(torch.clamp(-butterfly_loss, min=0))
        
        return calendar_loss, butterfly_loss


class ArbFreeDNN(nn.Module):
    def __init__(self, layers):
        super(ArbFreeDNN, self).__init__()
        self.ID = random.randint(0, 10000000)

        self.model = nn.Sequential(
            nn.Linear(121, layers[0]),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.25)
        )

        for idx in range(0, len(layers)-1):
            self.model.add_module(f'layer: {idx+1}', nn.Linear(layers[idx], layers[idx+1]))
            self.model.add_module(f'activation: {idx}', nn.LeakyReLU(0.1))
            self.model.add_module(f'dropout: ', nn.Dropout(0.25))

        self.model.add_module(f'last layer: ', nn.Linear(layers[-1], 121))
        self.model.add_module(f'softplus: ', nn.Softplus())

    def forward(self, x):
        out = self.model(x)
        return out
    
    def getID(self):
        return self.ID

def train_DNN(num_epochs, model, dataloader, optimizer, arb_mult, loss_fn=None, logging_interval=1, save_model=None):
    if loss_fn is None:
        #insert arb free loss here
        loss_re = F.mse_loss
        loss_arb = ArbFreeLoss()

    loss_chart = []
    re_loss_chart = []
    cal_loss_chart = []
    but_loss_chart = []

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        for idx, (feature, target) in enumerate(dataloader):
            feature.requires_grad = True
            output = model(feature)
            #batch_size = output.shape[0]

            re = loss_re(output, target, reduction='mean')
            cal, but = loss_arb(feature, output)
            loss = re + (arb_mult * (cal + but))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_chart.append(loss.item())
            re_loss_chart.append(re.item())
            cal_loss_chart.append(cal.item())
            but_loss_chart.append(but.item())

            if idx % logging_interval == 0:
                print(f'Epoch: {epoch} | Progress: {(idx / len(dataloader)):.2f}| Loss: {loss:.6f} | Time Elapsed: {((time.time() - start_time)/60):.2f} min')
        scheduler.step(loss)

    return loss_chart, re_loss_chart, cal_loss_chart, but_loss_chart

def eval_DNN(model, dataloader, loss_fn=None):
    if loss_fn is None:
        re_loss = F.mse_loss
        arb_loss = ArbFreeLoss()

    loss_chart_re = []
    loss_chart_cal = []
    loss_chart_but = []

    model.eval()
    for features, target in dataloader:
        features.requires_grad = True
        output = model(features)

        re = re_loss(output, target, reduction='mean')
        cal, but = arb_loss(features, output)

        loss_chart_re.append(re.item())
        loss_chart_cal.append(cal.item())
        loss_chart_but.append(but.item())

    return loss_chart_re, loss_chart_cal, loss_chart_but


########################
# Predictive Engine
########################

def MDALoss(original, future, preds):
    scores = []
    future = future.reshape(121)
    original = original.reshape(121)
    preds = preds.reshape(121)

    for idx in range(len(original)):
        true_dr = future[idx] - original[idx]
        pred_dr = preds[idx] - original[idx]
        if (true_dr > 0 and pred_dr > 0) or (true_dr < 0 and pred_dr < 0):
            scores.append(1)
        else:
            scores.append(0)
    return scores

def MAPEloss(future, preds):
    scores = []
    future = future.reshape(121)
    preds = preds.reshape(121)

    for idx in range(len(future)):
        p_term = abs(((future[idx] - preds[idx]) / future[idx]))
        scores.append(p_term)

    return scores

class PredictiveEngine():
    def __init__(self, trained_LSTM, trained_DNN, use_VAE, trained_VAE=None):
        self.vae = trained_VAE
        self.lstm = trained_LSTM
        self.dnn = trained_DNN
        self.use_VAE = use_VAE

    def validate(self, section, loss_fn=None):
        if loss_fn is None:
            loss_fn = F.mse_loss
            arb_fn = ArbFreeLoss()
        mse_chart = []
        rmse_chart = []
        mda_chart = []
        mape_chart = []
        cal_chart = []
        but_chart = []
        preds = []
        targets = []
        iv_data = getVAE_DataLoader(section='all', batch_size=1, scale=False, scaler_id=self.vae.getID())
        iv_data_subs = IVSDataForVAE(section=section, pkl_path='data/R2_STD_IVS_DFW_SORTED.pkl', scale=False, scaler_id=None, scaler_path=None)
        if self.use_VAE:
            data = getLSTM_Dataloader(section=section, vae_model=self.vae, iv_dataloader=iv_data, scale=True, NNtype='DNN', batch_size=2, full_ivs=False, scaler_id=self.lstm.getID())
        else:
            data = getLSTM_Dataloader(section=section, vae_model=None, iv_dataloader=iv_data, scale=True, NNtype='DNN', batch_size=2, full_ivs=True, scaler_id=self.lstm.getID())
        self.lstm.eval()
        self.dnn.eval()
        self.vae.eval()
        count = 1
        with torch.no_grad():
            for feature, _ in data:
                if count >= 510:
                    break
                og_feature = iv_data_subs[count-1]
                feature = feature[:, None, :]
                output = self.lstm(feature)
                if self.use_VAE:
                    '''
                    latent_means = []
                    latent_vars = []
                    for o_idx, tensor in enumerate(output):
                        latent_means.append([])
                        latent_vars.append([])
                        for idx, item in enumerate(tensor):
                            if idx % 2 == 0:
                                latent_means[o_idx].append(item.item())
                            else:
                                latent_vars[o_idx].append(item.item())
                    latent_means = torch.tensor(latent_means, dtype=torch.float32)
                    latent_vars = torch.tensor(latent_vars, dtype=torch.float32)
                    vae_out = self.vae.decode(latent_means, latent_vars)'''
                    vae_out = self.vae.decoder(output)
                    output = getUnscaledData(vae_out, f'scalers/VAE_scaler_{self.vae.getID()}.pkl')
                else:
                    output = getUnscaledData(output, f'scalers/LSTM_scaler_o_{self.lstm.getID()}.pkl')
                for i, out in enumerate(output):
                    with open(f'scalers/DNN_scaler_i_{self.dnn.getID()}.pkl', 'rb') as f:
                        dnn_i_scale = pickle.load(f)
                    dnn_in = dnn_i_scale.transform(out.reshape(1, 121))
                    dnn_in = torch.tensor(dnn_in, dtype=torch.float32)

                    with torch.enable_grad():
                        dnn_in.requires_grad = True
                        dnn_out = self.dnn(dnn_in)

                        cal, but = arb_fn(dnn_in, dnn_out)
                        cal_chart.append(cal.item())
                        but_chart.append(but.item())

                    dnn_out = getUnscaledData(dnn_out, f'scalers/DNN_scaler_o_{self.dnn.getID()}.pkl')

                    preds.append(dnn_out)
                    #target = getUnscaledData(target, f'scalers/LSTM_scaler_o_{self.lstm.getID()}.pkl')
                    targets.append(iv_data_subs[count])
                    mse = loss_fn(torch.tensor(dnn_out), torch.tensor(iv_data_subs[count].reshape(1, 121)), reduction='mean')

                    mse_chart.append(mse.item())
                    rmse_chart.append(sqrt(mse.item()))

                    #og_feature = getUnscaledData(og_feature, f'scalers/LSTM_scaler_i_{self.lstm.getID()}.pkl')
                    mda_chart.append(MDALoss(np.array(og_feature).reshape(1, 121), np.array(iv_data_subs[count]), dnn_out))
                    mape_chart.append(MAPEloss(np.array(iv_data_subs[count]), dnn_out))

                    count += 1

        return preds, targets, mse_chart, rmse_chart, mda_chart, mape_chart, cal_chart, but_chart
        
def getModels(vae_id, lstm_id, dnn_id):
    vae_models = pd.read_csv('model_scores/VAE.csv')
    lstm_models = pd.read_csv('model_scores/LSTM.csv')
    dnn_models = pd.read_csv('model_scores/DNN.csv')

    vae_ids = vae_models['ID#']
    lstm_ids = lstm_models['ID#']
    dnn_ids = dnn_models['ID#']
    vae = None
    if vae_id is not None:
        vae_idx = 0
        for idx in range(len(vae_ids)):
            if int(vae_id) == vae_ids.iloc[idx]:
                vae_idx = idx
        latent_dim = vae_models.iloc[vae_idx]['LATENT_DIM']
        num_layers = vae_models.iloc[vae_idx]['NUM_LAYERS']
        hidden_size = vae_models.iloc[vae_idx]['HIDDEN_SIZE']
        vae_type = vae_models.iloc[vae_idx]['VAE_TYPE']
        if vae_type == 'DNN':
            vae = DNNVAE(latent_dim=latent_dim, num_layers=num_layers, hidden_size=hidden_size)
            vae.load_state_dict(torch.load(f'all_models/VAE_{vae_id}.pt'))
        elif vae_type == 'CNN':
            vae = ConvVAE(latent_dim=latent_dim)
            vae.load_state_dict(torch.load(f'all_models/VAE_{vae_id}.pt'))
        vae.ID = int(vae_id)
    #############
    lstm = None
    if lstm_id is not None:
        lstm_idx = 0
        for idx in range(len(lstm_ids)):
            if int(lstm_id) == lstm_ids.iloc[idx]:
                lstm_idx = idx
        hidden_size = lstm_models.iloc[lstm_idx]['HIDDEN_SIZE']
        num_layers = lstm_models.iloc[lstm_idx]['NUM_LAYERS']
        latent_dim = lstm_models.iloc[lstm_idx]['LATENT_DIM']
        full_ivs = lstm_models.iloc[lstm_idx]['FULL_IVS']
        att = lstm_models.iloc[lstm_idx]['ATT']
        if att:
            lstm = AttentionLSTM(hidden_size=int(hidden_size), num_layers=int(num_layers), latent_dim=int(latent_dim), full_ivs=full_ivs)
        else:
            lstm = LSTM(hidden_size=int(hidden_size), num_layers=int(num_layers), latent_dim=int(latent_dim), full_ivs=full_ivs)
        lstm.load_state_dict(torch.load(f'all_models/LSTM_{lstm_id}.pt'))
        lstm.ID = int(lstm_id)
    #############
    dnn = None
    if dnn_id is not None:
        dnn_idx = 0
        for idx in range(len(dnn_ids)):
            if int(dnn_id) == dnn_ids.iloc[idx]:
                dnn_idx = idx
        layers = dnn_models.iloc[dnn_idx]['LAYERS']
        layers = literal_eval(layers)
        dnn = ArbFreeDNN(layers=layers)
        dnn.load_state_dict(torch.load(f'all_models/DNN_{dnn_id}.pt'))
        dnn.ID = int(dnn_id)

    return vae, lstm, dnn

def graphIVS(ivs):
    fig = plt.figure(40)
    ax = fig.add_subplot(111, projection='3d')

    date_using = pd.DataFrame(ivs)
    x, y = np.meshgrid(date_using.columns, date_using.index)
    z = date_using.values

    my_cmap = plt.get_cmap('viridis')
    ax.plot_surface(y, x, z, cmap=my_cmap)
    ax.view_init(20, 140) 
    ax.set_xlabel('Time to Expiration', labelpad=7)
    ax.set_ylabel('Moneyness', labelpad=6)
    ax.set_zlabel('Implied Volatility', labelpad=8)
    ax.set_box_aspect(aspect=None, zoom=0.7)
    plt.show()