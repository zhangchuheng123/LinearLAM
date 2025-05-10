from sklearn.cross_decomposition import CCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
            self.observation_pred = nn.Linear(d_o, d_o)
            self.noise_pred = nn.Linear(d_o, d_b)
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
            observation = self.observation_pred(obs_pred - o)
            noise = self.noise_pred(obs_pred - o)
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


def main(learn_A, CD_zero, pseudo_latent, use_kappa, no_noise_eval=True):

    N = 128         # batch size
    NN = 100000     # eval size
    do = 128        # dimension of observation 
    da = 8          # dimension of action 
    db = 128        # dimension of noise prediction

    sigma_list = [
        0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
        0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 
        1.2, 1.3, 1.4, 1.5,
    ]
    dz_list = [2, 4, 6, 8, 10, 12, 14, 16]

    record = []
    for sigma, dz in product(sigma_list, dz_list):

        print(f'sigma={sigma} da={da} dz={dz}')
        
        action_embeddings = np.random.randn(do, da)
        action_embeddings, _ = np.linalg.qr(action_embeddings) 
        pinv_action_embeddings = np.linalg.pinv(action_embeddings)

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
        for i_batch in range(10001):

            O = np.random.randn(do, N)
            A = np.random.randn(da, N)
            Q = action_embeddings @ A 
            noise = np.random.randn(do, N) * sigma

            Op = O + Q + noise
            # Checked: Var(O) = do   Var(Q) = da

            tensor_O = torch.tensor(O.T, dtype=torch.float32)
            tensor_Op = torch.tensor(Op.T, dtype=torch.float32)

            obs_pred, _ = lam(tensor_O, tensor_Op)
            loss = nn.MSELoss()(obs_pred, tensor_Op)
            opt1.zero_grad()
            loss.backward()
            opt1.step()

            if i_batch % 1000 == 0:
                # Evaluation
                for i_eval in range(5000):

                    O = np.random.randn(do, N)
                    A = np.random.randn(da, N)
                    Q = action_embeddings @ A 

                    if no_noise_eval:
                        Op = O + Q 
                        true_A = A
                    else:
                        noise = np.random.randn(do, N) * sigma
                        Op = O + Q + noise
                        true_A = pinv_action_embeddings @ (Q + noise)

                    tensor_O = torch.tensor(O.T, dtype=torch.float32)
                    tensor_Op = torch.tensor(Op.T, dtype=torch.float32)
                    target_A = torch.tensor(true_A.T, dtype=torch.float32)
                    target_O = torch.tensor(O.T, dtype=torch.float32)
                    if not no_noise_eval:
                        target_N = torch.tensor(noise.T, dtype=torch.float32)

                    _, preds = lam(tensor_O, tensor_Op)
                    act, obs, noi = preds
                    if no_noise_eval:
                        loss = nn.MSELoss()(act, target_A) + nn.MSELoss()(obs, target_O)
                    else:
                        loss = nn.MSELoss()(act, target_A) + nn.MSELoss()(obs, target_O) + nn.MSELoss()(noi, target_N)
                    opt2.zero_grad()
                    loss.backward()
                    opt2.step()

                if True:

                    O = np.random.randn(do, NN)
                    A = np.random.randn(da, NN)
                    Q = action_embeddings @ A 

                    if no_noise_eval:
                        Op = O + Q
                        true_A = A
                    else:
                        noise = np.random.randn(do, NN) * sigma
                        Op = O + Q + noise
                        true_A = pinv_action_embeddings @ (Q + noise)

                    tensor_O = torch.tensor(O.T, dtype=torch.float32)
                    tensor_Op = torch.tensor(Op.T, dtype=torch.float32)
                    tensor_A = torch.tensor(true_A.T, dtype=torch.float32)
                    if not no_noise_eval:
                        tensor_N = torch.tensor(noise.T, dtype=torch.float32)

                    obsp, preds = lam(tensor_O, tensor_Op)
                    act, obs, noi = preds

                    recon_loss = torch.mean(torch.sum(((obsp - tensor_Op) ** 2), axis=1)).item() / do
                    act_mse = torch.mean(torch.sum(((act - tensor_A) ** 2), axis=1)).item() / da
                    obs_mse = torch.mean(torch.sum(((obs - tensor_O) ** 2), axis=1)).item() / do
                    if sigma == 0 or no_noise_eval:
                        noi_mse = 1.0
                    else:
                        noi_mse = torch.mean(torch.sum(((noi - tensor_N) ** 2), axis=1)).item() / do / (sigma ** 2)

                record.append(dict(
                    do=do, da=da, dz=dz, sigma=sigma, iter=i_batch, 
                    recon_loss=recon_loss, act_mse=act_mse, obs_mse=obs_mse, noi_mse=noi_mse
                ))

                total_record = pd.DataFrame(record)
                total_record.to_csv(f'4_2_learnA{learn_A}_CDzero{CD_zero}_psdlatent{pseudo_latent}_nonoiseeval{no_noise_eval}.csv')

if __name__ == '__main__':
    main(learn_A=True, CD_zero=False, pseudo_latent=True, use_kappa=False, no_noise_eval=False)
