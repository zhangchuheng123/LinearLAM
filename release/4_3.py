import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from itertools import product


GET_ZERO_TENSOR = lambda shape: torch.tensor(np.zeros(shape), dtype=torch.float32)


class LAM_Linear(nn.Module):
    def __init__(self, d_o, d_z, d_a, d_b, pseudo_latent=True):
        super(LAM_Linear, self).__init__()
        self.pseudo_latent = pseudo_latent

        self.A = nn.Linear(d_o, d_o, bias=False)
        self.C = nn.Linear(d_o, d_z, bias=False)
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
        z = self.C(o + kappa1) + self.D(o_next + kappa1)
        obs_pred = self.A(o + kappa2) + self.B(z) - kappa2

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

def main(pseudo_latent=True):

    N = 128         # batch size
    NN = 100000     # eval size
    do = 128        # dimension of observation 
    da = 8          # dimension of action 
    db = 8          # dimension of noise prediction


    dz_list = list(range(2, 17))
    sigma_list = [0.5, 2.0]
    
    record = []
    for sigma, dz in product(sigma_list, dz_list):

        print(f'sigma={sigma} da={da} dz={dz}')

        action_embeddings = np.random.randn(do, da)
        action_embeddings, _ = np.linalg.qr(action_embeddings) 

        others_embeddings = np.random.randn(do, db)
        others_embeddings, _ = np.linalg.qr(others_embeddings)

        # Get model and optimizers
        lam = LAM_Linear(do, dz, da, db, learn_A=learn_A, CD_zero=CD_zero, pseudo_latent=pseudo_latent)
        opt1 = optim.Adam(get_parameters(lam.A, lam.B, lam.C, lam.D))
        opt2 = optim.Adam(get_parameters(lam.action_pred, lam.observation_pred, lam.noise_pred))

        # Training
        for i_batch in range(40001):

            O = np.random.randn(do, N)
            A = np.random.randn(da, N)
            Q = action_embeddings @ A 
            B = np.random.randn(db, N)
            noise = others_embeddings @ B * sigma

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
                    B = np.random.randn(db, N)
                    noise = others_embeddings @ B * sigma
                    Op = O + Q + noise

                    tensor_O = torch.tensor(O.T, dtype=torch.float32)
                    tensor_Op = torch.tensor(Op.T, dtype=torch.float32)
                    target_A = torch.tensor(A.T, dtype=torch.float32)
                    target_O = torch.tensor(O.T, dtype=torch.float32)
                    target_B = torch.tensor(B.T, dtype=torch.float32)

                    _, preds = lam(tensor_O, tensor_Op)
                    act, obs, noi = preds
                    loss = nn.MSELoss()(act, target_A) + nn.MSELoss()(obs, target_O) + nn.MSELoss()(noi, target_B)
                    opt2.zero_grad()
                    loss.backward()
                    opt2.step()

                if True:
                    O = np.random.randn(do, N)
                    A = np.random.randn(da, N)
                    Q = action_embeddings @ A 
                    B = np.random.randn(db, N)
                    noise = others_embeddings @ B * sigma
                    Op = O + Q + noise
                    # Op = O + Q

                    tensor_O = torch.tensor(O.T, dtype=torch.float32)
                    tensor_Op = torch.tensor(Op.T, dtype=torch.float32)
                    tensor_A = torch.tensor(A.T, dtype=torch.float32)
                    tensor_B = torch.tensor(B.T, dtype=torch.float32)

                    obsp, preds = lam(tensor_O, tensor_Op)
                    act, obs, noi = preds

                    recon_loss = torch.mean(torch.sum(((obsp - tensor_Op) ** 2), axis=1)).item() / do
                    act_mse = torch.mean(torch.sum(((act - tensor_A) ** 2), axis=1)).item() / da
                    obs_mse = torch.mean(torch.sum(((obs - tensor_O) ** 2), axis=1)).item() / do
                    if sigma == 0:
                        noi_mse = 1.0
                    else:
                        noi_mse = torch.mean(torch.sum(((noi - tensor_B) ** 2), axis=1)).item() / db

                record.append(dict(
                    do=do, da=da, dz=dz, db=db, sigma=sigma, iter=i_batch, 
                    recon_loss=recon_loss, act_mse=act_mse, obs_mse=obs_mse, noi_mse=noi_mse
                ))

                total_record = pd.DataFrame(record)
                total_record.to_csv(f'4_3.csv')


if __name__ == '__main__':
    main(pseudo_latent=True)
