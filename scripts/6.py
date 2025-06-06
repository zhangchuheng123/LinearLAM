from sklearn.cross_decomposition import CCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import product


GET_ZERO_TENSOR = lambda shape: torch.tensor(np.zeros(shape), dtype=torch.float32)


class LAM_Linear(nn.Module):
    def __init__(self, d_o, d_z, d_a, d_b, learn_A=True, CD_zero=True, pseudo_latent=True):
        super(LAM_Linear, self).__init__()
        self.learn_A = learn_A
        self.CD_zero = CD_zero
        self.pseudo_latent = pseudo_latent

        if self.learn_A:
            self.A = nn.Linear(d_o, d_o, bias=False)
        self.C = nn.Linear(d_o, d_z, bias=False)
        if not self.CD_zero:
            self.D = nn.Linear(d_o, d_z, bias=False)
        self.B = nn.Linear(d_z, d_o, bias=False)
        if self.pseudo_latent:
            self.action_pred = nn.Linear(d_o, d_a)
        else:
            self.action_pred = nn.Linear(d_z, d_a)
        self.observation_pred = nn.Linear(d_z, d_o)
        self.noise_pred = nn.Linear(d_z, d_b)

    def forward(self, o, o_next, kappa1=None, kappa2=None):
        if kappa1 is None:
            kappa1 = GET_ZERO_TENSOR(o.shape)
        if kappa2 is None:
            kappa2 = GET_ZERO_TENSOR(o.shape)
        if self.CD_zero:
            z = self.C(o + kappa1) - self.C(o_next + kappa1)
        else:
            z = self.C(o + kappa1) + self.D(o_next + kappa1)
        if self.learn_A:
            obs_pred = self.A(o + kappa2) + self.B(z) - kappa2
        else:
            obs_pred = (o + kappa2) + self.B(z) - kappa2

        if self.pseudo_latent:
            action = self.action_pred(obs_pred - o)
        else:
            action = self.action_pred(z)
        observation = self.observation_pred(z)
        noise = self.noise_pred(z)
        return obs_pred, (action, observation, noise)


def get_parameters(*args):
    res = []
    for p in args:
        res = res + list(p.parameters())
    return res

def main(learn_A, CD_zero, pseudo_latent):

    N = 128
    NN = 10000
    do = 128
    da = 8
    db = 128    # dimension of noise prediction

    dz_list = list(range(2, 17))
    kappa_coeff_list = [0, 0.01, 0.1, 0.5]       # magnitude of kappa
    sigma_list = [0, 0.1, 0.5, 1.0]
    
    # existing = pd.read_csv('5_learnATrue_CDzeroFalse_psdactionTrue_morechi_evalrand.csv', index_col=0)
    # record = existing.to_dict('records')
    record = []
    tic = time.time()

    for sigma, dz, kappa_coeff in product(sigma_list, dz_list, kappa_coeff_list):

        print(f'noise sigma={sigma} kappa_coeff={kappa_coeff} dz={dz} time={time.time() - tic:.2f}s')
        tic = time.time()
        # matching_record = [item for item in record if item['sigma'] == sigma and item['dz'] == dz and item['kappa_coeff'] == kappa_coeff and item['iter'] == 40000]

        # if len(matching_record) > 0:
        #     continue

        action_embeddings = np.random.randn(do, da)
        action_embeddings, _ = np.linalg.qr(action_embeddings) 

        # Get model and optimizers
        lam = LAM_Linear(do, dz, da, db, learn_A=learn_A, CD_zero=CD_zero, pseudo_latent=pseudo_latent)
        if CD_zero:
            if learn_A:
                opt1 = optim.Adam(get_parameters(lam.A, lam.B, lam.C))
            else:
                opt1 = optim.Adam(get_parameters(lam.B, lam.C))
        else:
            if learn_A:
                opt1 = optim.Adam(get_parameters(lam.A, lam.B, lam.C, lam.D))
            else:
                opt1 = optim.Adam(get_parameters(lam.B, lam.C, lam.D))
        opt2 = optim.Adam(get_parameters(lam.action_pred, lam.observation_pred, lam.noise_pred))

        # Training
        for i_batch in range(40001):

            O = np.random.randn(do, N)
            A = np.random.randn(da, N)
            Q = action_embeddings @ A 
            noise = np.random.rand(do, N) * sigma

            kappa1 = torch.tensor(np.random.randn(N, do) * kappa_coeff, dtype=torch.float32)
            kappa2 = torch.tensor(np.random.randn(N, do) * kappa_coeff, dtype=torch.float32)

            Op = O + Q + noise
            # Checked: Var(O) = do   Var(Q) = da

            tensor_O = torch.tensor(O.T, dtype=torch.float32)
            tensor_Op = torch.tensor(Op.T, dtype=torch.float32)

            obs_pred, _ = lam(tensor_O, tensor_Op, kappa1, kappa2)
            loss = nn.MSELoss()(obs_pred, tensor_Op)
            opt1.zero_grad()
            loss.backward()
            opt1.step()

        # Evaluation
        for i_eval in range(5000):

            O = np.random.randn(do, N)
            # A = (1 - chi) * np.random.randn(da, N) + chi * policy_embeddings @ O
            A = np.random.randn(da, N)
            Q = action_embeddings @ A 
            noise = np.random.rand(do, N) * sigma

            Op = O + Q + noise

            tensor_O = torch.tensor(O.T, dtype=torch.float32)
            tensor_Op = torch.tensor(Op.T, dtype=torch.float32)
            target_A = torch.tensor(A.T, dtype=torch.float32)
            target_O = torch.tensor(O.T, dtype=torch.float32)
            target_N = torch.tensor(noise.T, dtype=torch.float32)

            _, preds = lam(tensor_O, tensor_Op)
            act, obs, noi = preds
            loss = nn.MSELoss()(act, target_A) + nn.MSELoss()(obs, target_O) + nn.MSELoss()(noi, target_N)
            opt2.zero_grad()
            loss.backward()
            opt2.step()

        if True:

            O = np.random.randn(do, NN)
            # A = (1 - chi) * np.random.randn(da, NN) + chi * policy_embeddings @ O
            A = np.random.randn(da, NN)
            Q = action_embeddings @ A 
            noise = np.random.rand(do, NN) * sigma

            Op = O + Q + noise

            tensor_O = torch.tensor(O.T, dtype=torch.float32)
            tensor_Op = torch.tensor(Op.T, dtype=torch.float32)
            target_A = torch.tensor(A.T, dtype=torch.float32)
            target_O = torch.tensor(O.T, dtype=torch.float32)
            target_N = torch.tensor(noise.T, dtype=torch.float32)
            obsp, preds = lam(tensor_O, tensor_Op)
            act, obs, noi = preds

            recon_loss = torch.mean(torch.sum(((obsp - tensor_Op) ** 2), axis=1)).item() / do
            act_mse = torch.mean(torch.sum(((act - target_A) ** 2), axis=1)).item() / da
            obs_mse = torch.mean(torch.sum(((obs - target_O) ** 2), axis=1)).item() / do
            if sigma > 0:
                noi_mse = torch.mean(torch.sum(((noi - target_N) ** 2), axis=1)).item() / do / (sigma ** 2)
            else:
                noi_mse = 1.0

        record.append(dict(
            do=do, da=da, dz=dz, db=db, sigma=sigma, iter=i_batch, kappa_coeff=kappa_coeff,
            recon_loss=recon_loss, act_mse=act_mse, obs_mse=obs_mse, noi_mse=noi_mse))

        total_record = pd.DataFrame(record)
        total_record.to_csv(f'6_learnA{learn_A}_CDzero{CD_zero}_psdaction{pseudo_latent}.csv')

if __name__ == '__main__':
    main(learn_A=True, CD_zero=False, pseudo_latent=False)

