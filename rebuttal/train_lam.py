import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange
from tqdm import tqdm
import torch.nn.functional as F

# this script provides a minimal implementation of the LAM model
# trained on the dataset created by data_create_lam.py
# the model is trained to reconstruct the next frame in the movie
# given the current frame and the next frame

data_path_clean = 'dataset/noise_0.25/'
data_path = 'dataset/noise_0.25/'
# data_path = '/home/timpearce/02_lam/dataset_0.95/'
# data_path = '/home/timpearce/02_lam/dataset_0.99/'
# data_path = '/home/timpearce/02_lam/dataset_01/'

# save_path_viz = '/home/timpearce/02_lam/01_visualizations_02/' # action sweep good
# save_path_viz = '/home/timpearce/02_lam/01_visualizations_03_intensity/' # trying rerun over intensity
# save_path_viz = '/home/timpearce/02_lam/01_visualizations_04_debug/' # try to get 5 codes working
# save_path_viz = '/home/timpearce/02_lam/01_visualizations_05_action/' # rerunning for action prop test 
# save_path_viz = '/home/timpearce/02_lam/01_visualizations_06_action/' # rerunning for action prop test, trying mlp
# save_path_viz = '/home/timpearce/02_lam/01_visualizations_07_cleanrun/' 
# save_path_viz = '/home/timpearce/02_lam/01_visualizations_08_cleanrun_lowcommit/' 
# save_path_viz = '/home/timpearce/02_lam/01_visualizations_08_cleanrun_lowcommit_dataset0.95/' 
# # save_path_viz = '/home/timpearce/02_lam/01_visualizations_08_cleanrun_lowcommit_dataset0.99/' 
# save_path_viz = '/home/timpearce/02_lam/01_visualizations_09/' 
save_path_viz = 'results/noise_0.25/' 

os.makedirs(save_path_viz)

n_latent = 16 # was 16
# codebook_size = 2048
# intensity = 1.0 # noise intensity
# action_prop = 0.9 # proportion of traj that have action labels allowed
# k_data_aug = 0.1 # data augmentation factor, 0.0 means no augmentation
device = 'cuda'

# for intensity in [0.0, 0.25, 0.5, 0.75, 1.0]:
# for intensity in [1.0, 2.0]:
# for intensity in [0.0, 1.0, 2.0, 4.0]: # _03
# for intensity in [0.0, 4.0, 8.0]: # _04
for intensity in [0.25]: # _05
    # for codebook_size in [2,3,4,5,6,8,10,20,40,80]:
    # for codebook_size in [4, 8, 16, 64, 256, 512, 1024, 2048]: # _03
    # for codebook_size in [1024]: # _04
    for codebook_size in [5]: # _05
    # for codebook_size in [4,5]:
    # for codebook_size in [4, 8, 16, 64, 256, 512, 1024, 2048, 4096]:
        # for action_prop in [1.0, 0.1, 0.01, 0.001, 0.0]: # _05
        # for action_prop in [0.0, 0.001, 0.01, 0.1, 1.0]: # _06
        # for action_prop in [0.0, 0.001, 0.01, 0.05, 0.1, 1.0]:
        for action_prop in [0.0]:
            # for k_data_aug in [0., 0.1]:
            for k_data_aug in [0.0]:

                print('intensity', intensity)
                print('codebook_size', codebook_size)
                print('action_prop', action_prop)
                print('k_data_aug', k_data_aug)
                # dataloader
                class LAM_Dataset(torch.utils.data.Dataset):
                    def __init__(self, data_path, n_traj, n_frames, intensity=1.0, action_prop=1.0):
                        self.data_path = data_path
                        self.n_traj = n_traj
                        self.n_frames = n_frames
                        self.intensity = intensity
                        self.action_prop = action_prop
                        self.action_yes = int(self.n_traj * self.action_prop)

                    def __len__(self):
                        return self.n_traj

                    def __getitem__(self, idx, frame_i=None):
                        x = np.load(self.data_path + 'x_'+str(idx)+'.npy')
                        x = torch.from_numpy(x).float()
                        x[:, 3:, :] = x[:, 3:, :] * intensity # change intensity of the noise
                        a = np.load(self.data_path + 'a_'+str(idx)+'.npy')
                        a = torch.from_numpy(a).float()
                        if frame_i is None:
                            # now subselect one frame and the next frame
                            frame_i = np.random.randint(0, self.n_frames-1)
                        x_curr = x[frame_i]
                        x_next = x[frame_i+1]
                        a_curr = a[frame_i]
                        if idx < self.action_yes:
                            a_allowed = torch.tensor([1.0])
                        else:
                            a_allowed = torch.tensor([0.0])
                        return x_curr, x_next, a_curr, a_allowed

                # model 1, very small CNN encoder, taking shape 4x4 as input, outputing 2x2
                class Encoder(torch.nn.Module):
                    def __init__(self):
                        super(Encoder, self).__init__()
                        n_filters = 64 # was 64
                        self.conv1 = torch.nn.Conv2d(1, n_filters, 3, padding=1)
                        self.bn1 = torch.nn.BatchNorm2d(n_filters)
                        self.conv2 = torch.nn.Conv2d(n_filters, n_filters, 3, padding=1)
                        self.bn2 = torch.nn.BatchNorm2d(n_filters)
                        self.conv3 = torch.nn.Conv2d(n_filters, 1, 3, padding=1)
                        self.pool = torch.nn.AvgPool2d(2)
                        self.relu = torch.nn.ReLU()

                    def forward(self, x):
                        x = self.relu(self.conv1(x))
                        x = self.bn1(x)
                        x = self.relu(self.conv2(x))
                        x = self.bn2(x)
                        x = self.pool(x)
                        x = self.relu(self.conv3(x))
                        # flatten
                        x = x.view(x.size(0), -1)
                        return x

                # alternative model 1, small fully connected mlp, taking shape 4x4x2 as input, outputing 2x2x2
                # class Encodermlp(torch.nn.Module):
                #     def __init__(self):
                #         super(Encodermlp, self).__init__()
                #         self.fc1 = torch.nn.Linear(4*4*2, 512)
                #         self.fc2 = torch.nn.Linear(512, 512)
                #         self.fc3 = torch.nn.Linear(512, 512)
                #         self.fc4 = torch.nn.Linear(512, 8)
                #         self.relu = torch.nn.ReLU()
                #     def forward(self, x):
                #         x = x.view(x.size(0), -1)
                #         x = self.relu(self.fc1(x))
                #         x = self.relu(self.fc2(x))
                #         x = self.relu(self.fc3(x))
                #         x = self.fc4(x)
                #         return x

                # model 2, small MLP, taking the concatenation of the two encodings as input
                class VQMLP(torch.nn.Module):
                    def __init__(self, n_latent, codebook_size):
                        super(VQMLP, self).__init__()
                        self.fc1 = torch.nn.Linear(2*4, 512)
                        self.fc1_extra = torch.nn.Linear(512, 512)
                        self.fc2 = torch.nn.Linear(512, n_latent)
                        self.relu = torch.nn.ReLU()
                        self.codebook = torch.randn(codebook_size, n_latent, requires_grad=True)
                    
                    def forward(self, x):
                        x = self.relu(self.fc1(x))
                        x = self.relu(self.fc1_extra(x)) # optional extra layer
                        x = self.fc2(x)

                        # VQ
                        distances = torch.cdist(x, self.codebook)
                        nearest_indices = torch.argmin(distances, dim=1)
                        x_quantized = self.codebook[nearest_indices]
                        # x_quantized = x # don't use quantized for now
                        x_quantized = x + (x_quantized - x).detach() # straight-through estimator
                        # quantization loss
                        loss_quant = torch.mean((self.codebook[nearest_indices] - x.detach())**2)
                        # commitment loss
                        loss_commit = torch.mean((x_quantized.detach() - x)**2)

                        # Entropy loss to encourage diverse code usage
                        one_hot = F.one_hot(nearest_indices, num_classes=self.codebook.size(0)).float()
                        avg_probs = one_hot.mean(dim=0)
                        entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
                        loss_entropy = - entropy  # tune weight
                        return x, x_quantized, nearest_indices, loss_quant, loss_commit, loss_entropy

                # model 3, small unet, condition on x_curr and diff_quant, try to predict x_next
                class Unet(torch.nn.Module):
                    def __init__(self):
                        super(Unet, self).__init__()
                        n_filters = 64 # 64
                        self.conv1 = torch.nn.Conv2d(2, n_filters//1, 3, padding=1)
                        self.conv2 = torch.nn.Conv2d(n_filters//1, n_filters, 3, padding=1)
                        self.conv3 = torch.nn.Conv2d(n_filters, n_filters, 3, padding=1)
                        self.conv4 = torch.nn.Conv2d(n_filters, n_filters//1, 3, padding=1)
                        self.conv5 = torch.nn.Conv2d(n_filters//1, 1, 3, padding=1)
                        self.pool = torch.nn.AvgPool2d(2)
                        self.upsample = torch.nn.UpsamplingNearest2d(scale_factor=(2,2))
                        self.relu = torch.nn.ReLU()
                        # batchnorm
                        self.bn1 = torch.nn.BatchNorm2d(n_filters//1)
                        self.bn2 = torch.nn.BatchNorm2d(n_filters)
                        self.bn3 = torch.nn.BatchNorm2d(n_filters)
                        self.bn4 = torch.nn.BatchNorm2d(n_filters//1)
                        # self.bn5 = torch.nn.BatchNorm

                    def forward(self, x):
                        x = self.relu(self.conv1(x))
                        x = self.bn1(x)
                        x = self.relu(self.conv2(x))
                        x = self.bn2(x)
                        x = self.pool(x)
                        x = self.relu(self.conv3(x))
                        x = self.bn3(x)
                        x = self.upsample(x) # needs N C H W
                        x = self.relu(self.conv4(x))
                        x = self.bn4(x)
                        x = self.relu(self.conv5(x))
                        return x

                def reconstruct(x_curr, x_next, a_curr, save_name='recon_frame'):
                    # visualize the reconstructions
                    with torch.no_grad():
                        model_encoder.eval()
                        model_mlp.eval()
                        model_unet.eval()
                        
                        x_curr = x_curr.unsqueeze(0).unsqueeze(1)
                        x_next = x_next.unsqueeze(0).unsqueeze(1)
                        x_curr_enc = model_encoder(x_curr)
                        x_next_enc = model_encoder(x_next)
                        x_cat = torch.cat([x_curr_enc, x_next_enc], dim=1)
                        x_cat = x_cat.view(x_cat.size(0), -1)
                        diff, diff_quant, diff_idx, loss_quant, loss_commit, loss_entropy = model_mlp(x_cat)
                        diff_quant_img = diff_quant.view(-1, 1, 4, 4)
                        x_next_hat = model_unet(torch.cat([x_curr, diff_quant_img], dim=1))
                        x_curr = rearrange(x_curr, 'B C H W -> B H W C')
                        x_next = rearrange(x_next, 'B C H W -> B H W C')
                        x_next_hat = rearrange(x_next_hat, 'B C H W -> B H W C')

                        fig, axs = plt.subplots(1, 3, figsize=(15,5))
                        axs[0].imshow(x_curr[0].cpu(), cmap='gray')
                        axs[0].set_title('x_curr')
                        axs[1].imshow(x_next[0].cpu(), cmap='gray')
                        axs[1].set_title('x_next '+str(a_curr))
                        axs[2].imshow(x_next_hat[0].cpu(), cmap='gray')
                        axs[2].set_title('x_next_hat')
                        fig.savefig(save_path_viz + save_name)
                        plt.close(fig)

                        model_encoder.train()
                        model_mlp.train()
                        model_unet.train()
                    return None

                # training loop
                n_epochs = 400 # was 400
                batch_size = 1024
                l_rate=1e-4
                enc_type = 'cnn' # 'cnn' or 'mlp'

                model_encoder = Encoder().to(device)
                # model_encoder_mlp = Encodermlp().to(device)
                model_mlp = VQMLP(n_latent, codebook_size).to(device)
                model_mlp.codebook = model_mlp.codebook.to(device)
                model_unet = Unet().to(device)
                optimizer = torch.optim.Adam(list(model_encoder.parameters()) + list(model_mlp.parameters()) + list(model_unet.parameters()), lr=l_rate)
                bad_code_list = {}

                # print model sizes
                print('model_encoder', sum(p.numel() for p in model_encoder.parameters()))
                # print('model_encoder_mlp', sum(p.numel() for p in model_encoder_mlp.parameters()))
                print('model_mlp', sum(p.numel() for p in model_mlp.parameters()))
                print('model_unet', sum(p.numel() for p in model_unet.parameters()))

                # make proper dataloader
                dataset = LAM_Dataset(data_path, n_traj=4000, n_frames=200, intensity=intensity, action_prop=action_prop)
                # divide into train and test
                train_size = int(0.95 * len(dataset))
                test_size = len(dataset) - train_size
                train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
                dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                # dataloader_val = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

                # dataset without correlations
                dataset_clean = LAM_Dataset(data_path_clean, n_traj=4000, n_frames=200, intensity=intensity, action_prop=action_prop)
                train_size_clean = int(0.95 * len(dataset_clean))
                test_size_clean = len(dataset_clean) - train_size_clean
                train_dataset_clean, test_dataset_clean = torch.utils.data.random_split(dataset_clean, [train_size, test_size])
                dataloader_clean = torch.utils.data.DataLoader(train_dataset_clean, batch_size=batch_size, shuffle=True)

                model_encoder.train()
                # model_encoder_mlp.train()
                model_mlp.train()
                model_unet.train()
                x_curr_test, x_next_test, a_curr_test, a_allowed_test = dataset.__getitem__(0, frame_i=3)
                loss_all = []
                for epoch in range(n_epochs):
                    losses = []
                    for x_curr, x_next, a_curr, a_allowed in tqdm(dataloader):
                        # add channel dim
                        x_curr = x_curr.unsqueeze(1).to(device)
                        x_next = x_next.unsqueeze(1).to(device)
                        a_curr = a_curr.to(device)

                        # data augmentation
                        # x_k1 = -x_curr.clone()
                        # x_k2 = -x_curr.clone()
                        # x_k1[:,:,:3,:] = 0.
                        # x_k2[:,:,:3,:] = 0.
                        # # x_k1[:,:,:3,:] = torch.clamp(torch.randn_like(x_curr[:,:,3:,:])+0.5, 0, 1)
                        # # x_k2[:,:,:3,:] = torch.clamp(torch.randn_like(x_next[:,:,3:,:])+0.5, 0, 1)
                        # x_k1[:,:,3:,:] = torch.rand_like(x_curr[:,:,3:,:])
                        # x_k2[:,:,3:,:] = torch.rand_like(x_next[:,:,3:,:])


                        # x_k1 = x_k1.to(device)
                        # x_k2 = x_k2.to(device)

                        if enc_type == 'cnn':
                            if k_data_aug > 0.0:
                                x_curr_in = x_curr.clone()
                                x_next_in = x_next.clone()
                                # we should sample a new fresh noise mask, but this is easier for now and same effect
                                x_curr_in[:,:,3:,:] = x_next_in[:,:,3:,:]
                            else:
                                x_curr_in = x_curr
                                x_next_in = x_next
                            x_curr_enc = model_encoder(x_curr_in)
                            x_next_enc = model_encoder(x_next_in)
                            x_cat = torch.cat([x_curr_enc, x_next_enc], dim=1)
                            x_cat = x_cat.view(x_cat.size(0), -1)
                        # elif enc_type == 'mlp':
                        #     x_input = torch.cat([x_curr + k_data_aug*x_k1, x_next + k_data_aug*x_k2], dim=1)
                        #     x_input = x_input.view(x_input.size(0), -1)
                        #     # x_cat = model_encoder_mlp(x_input)
                        diff, diff_quant, diff_idx, loss_quant, loss_commit, loss_entropy = model_mlp(x_cat)

                        # now get diff_quant from looking up a_curr directly
                        diff_quant_mix = diff_quant.clone()
                        if codebook_size >= 5:
                            diff_quant_actions = model_mlp.codebook[a_curr.long()][:,0,:].detach() # could detach
                            diff_quant_mix[a_allowed[:,0] == 1] = diff_quant_actions[a_allowed[:,0] == 1]

                        # additional loss pulling diff_quant towards diff_quant_actions for a_allowed
                        if codebook_size >= 5:
                            loss_act_pull = torch.mean((diff_quant[a_allowed[:,0] == 1] - diff_quant_actions[a_allowed[:,0] == 1])**2)
                        else:
                            loss_act_pull = torch.tensor(0.0)

                        diff_quant_img = diff_quant_mix.view(-1, 1, 4, 4)
                        if k_data_aug > 0.0:
                            x_curr_in = x_curr.clone()
                            x_next_in = x_next.clone()
                            x_next_in[:,:,3:,:] = x_curr_in[:,:,3:,:] # have to do this the other way
                        else:
                            x_curr_in = x_curr
                            x_next_in = x_next
                        x_next_hat = model_unet(torch.cat([x_curr_in, diff_quant_img], dim=1))
                        loss_recon = torch.mean((x_next_hat - x_next_in)**2)
                        # with torch.no_grad(): # to be comparable with other non aug models, would need to do this
                            # x_next_hat_clean = model_unet(torch.cat([x_curr, diff_quant_img], dim=1))

                        # loss = loss_recon
                        loss = loss_recon + 1.0/10*loss_quant + 0.25/10*loss_commit + 1.0/10*loss_act_pull # + 0.1*loss_entropy
                        loss_recon_ctrlable = torch.mean((x_next_hat[:,:,:3,:] - x_next_in[:,:,:3,:])**2)
                        loss_recon_stochastic = torch.mean((x_next_hat[:,:,3:,:] - x_next_in[:,:,3:,:])**2)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        # compute entropy of each codebook according to a_curr.long()[:,0]
                        mean_entropy = 0.0
                        for i in range(codebook_size):
                            mask_i = diff_idx==i
                            a_curr_i = a_curr.long()[mask_i,0]
                            # now compute entropy of a_curr_i
                            # turn this into probabilities of each item in a_curr_i
                            if len(a_curr_i) > 0:
                                counts = torch.bincount(a_curr_i, minlength=5)
                                
                                probs = counts.float() / len(a_curr_i)
                                probs = probs[probs > 0]
                                mean_entropy += -torch.sum(probs * torch.log(probs))
                            else:
                                mean_entropy += 0.0
                        mean_entropy = mean_entropy.item()/codebook_size

                        #=======#=======#=======#=======
                        # now have to do a batch from the clean dataset without data aug
                        if '0.' in data_path: # or k_data_aug > 0.:
                            with torch.no_grad():
                                for x_curr, x_next, a_curr, a_allowed in tqdm(dataloader_clean):
                                    # print(x_curr[0])
                                    # print(x_next[0])
                                    # print(a_curr[0])

                                    # add channel dim
                                    x_curr = x_curr.unsqueeze(1).to(device)
                                    x_next = x_next.unsqueeze(1).to(device)
                                    a_curr = a_curr.to(device)

                                    # data augmentation
                                    # x_k1 = x_curr.clone()*0.
                                    # x_k2 = x_next.clone()*0.
                                    # # x_k1[:,:,:3,:] = torch.clamp(torch.randn_like(x_curr[:,:,3:,:])+0.5, 0, 1)
                                    # # x_k2[:,:,:3,:] = torch.clamp(torch.randn_like(x_next[:,:,3:,:])+0.5, 0, 1)
                                    # x_k1[:,:,3:,:] = torch.rand_like(x_curr[:,:,3:,:])
                                    # x_k2[:,:,3:,:] = torch.rand_like(x_next[:,:,3:,:])
                                    # x_k1 = x_k1.to(device)
                                    # x_k2 = x_k2.to(device)

                                    if enc_type == 'cnn':
                                        # x_curr_enc = model_encoder(x_curr + k_data_aug*x_k1)
                                        # x_next_enc = model_encoder(x_next + k_data_aug*x_k1)
                                        x_curr_enc = model_encoder(x_curr)
                                        x_next_enc = model_encoder(x_next)
                                        x_cat = torch.cat([x_curr_enc, x_next_enc], dim=1)
                                        x_cat = x_cat.view(x_cat.size(0), -1)
                                    elif enc_type == 'mlp':
                                        x_input = torch.cat([x_curr + k_data_aug*x_k1, x_next + k_data_aug*x_k2], dim=1)
                                        x_input = x_input.view(x_input.size(0), -1)
                                        # x_cat = model_encoder_mlp(x_input)
                                    diff, diff_quant, diff_idx, loss_quant, loss_commit, loss_entropy = model_mlp(x_cat)

                                    # now get diff_quant from looking up a_curr directly
                                    diff_quant_mix = diff_quant.clone()
                                    if codebook_size >= 5:
                                        diff_quant_actions = model_mlp.codebook[a_curr.long()][:,0,:].detach() # could detach
                                        diff_quant_mix[a_allowed[:,0] == 1] = diff_quant_actions[a_allowed[:,0] == 1]

                                    # additional loss pulling diff_quant towards diff_quant_actions for a_allowed
                                    if codebook_size >= 5:
                                        loss_act_pull = torch.mean((diff_quant[a_allowed[:,0] == 1] - diff_quant_actions[a_allowed[:,0] == 1])**2)
                                    else:
                                        loss_act_pull = torch.tensor(0.0)

                                    diff_quant_img = diff_quant_mix.view(-1, 1, 4, 4)
                                    x_next_hat = model_unet(torch.cat([x_curr, diff_quant_img], dim=1))
                                    loss_recon = torch.mean((x_next_hat - x_next)**2)
                                    # with torch.no_grad(): # to be comparable with other non aug models, would need to do this
                                        # x_next_hat_clean = model_unet(torch.cat([x_curr, diff_quant_img], dim=1))

                                    # loss = loss_recon
                                    loss = loss_recon + 1.0/10*loss_quant + 0.25/10*loss_commit + 1.0/10*loss_act_pull # + 0.1*loss_entropy
                                    loss_recon_ctrlable = torch.mean((x_next_hat[:,:,:3,:] - x_next[:,:,:3,:])**2)
                                    loss_recon_stochastic = torch.mean((x_next_hat[:,:,3:,:] - x_next[:,:,3:,:])**2)
                                    break

                        #=======#=======#=======#=======#=======


                        losses.append([loss_recon.item(), loss_quant.item(), loss_commit.item(), loss_recon_ctrlable.item(), loss_recon_stochastic.item(), loss_act_pull.item()])
                    np_losses = np.array(losses)
                    loss_all.append(np.mean(np_losses, axis=0))
                    print('epoch', epoch, 'losses', np.mean(losses, axis=0))
                    print('\npurity, mean_entropy', round(mean_entropy,3))
                    if epoch % 10 == 0:
                        print('\nprobs', counts.float() / len(a_curr_i))

                    if epoch % 50 == 0:
                        reconstruct(x_curr_test.clone().to(device), x_next_test.clone().to(device), a_curr_test.clone().to(device), save_name='recon_frame_'+str(epoch)+'.png')

                    if epoch % 5 == 0:
                    # if True:
                        print(counts)

                        fig, axs = plt.subplots(1, 4, figsize=(20,5))
                        axs[0].plot(np.array(loss_all)[:,0])
                        axs[0].set_title('loss_recon')
                        axs[0].grid()
                        axs[1].plot(np.array(loss_all)[:,3])
                        axs[1].set_title('loss_recon_ctrlable')
                        axs[1].grid()
                        axs[2].plot(np.array(loss_all)[:,4])
                        axs[2].set_title('loss_recon_stochastic')
                        axs[2].grid()
                        axs[3].plot(np.array(loss_all)[:,1])
                        axs[3].set_title('loss_quant commit')
                        axs[3].grid()
                        fig.savefig(save_path_viz + 'losses_codes'+str(codebook_size)+'_intensity'+str(intensity)+'_actionprop'+str(action_prop)+'_k_data_aug'+str(k_data_aug)+'.png')
                        plt.close(fig)

                        # also print stats of codebook usage for each codebook
                        # print('\n==codebook usage:')
                        if action_prop == 0.0:
                            # if epoch % 10 == 0:
                            #     # first time, reinit all around largest one
                            #     for i in range(codebook_size):
                            #         if torch.mean((diff_idx == i)*1.).item() > 0.5:
                            #             print('reinitializing codebook', i, 'usage', round(torch.mean((diff_idx == i)*1.).item(),3))
                            #             for j in range(codebook_size):
                            #                 # if j != i:
                            #                 with torch.no_grad():
                            #                     # model_mlp.codebook[j] = model_mlp.codebook[i].clone() + torch.randn(n_latent, requires_grad=True).to(device) * 0.0001
                            #                     model_mlp.codebook[j] = diff[j] # reinit to an actual processed embedding (pre quantization)
                            #                 # break

                            # if epoch % 10 == 0:
                            # first time, reinit all around largest one
                            for i in range(codebook_size):
                                if torch.mean((diff_idx == i)*1.).item() < 0.001:
                                    if i not in bad_code_list:
                                        bad_code_list[i] = 0
                                    bad_code_list[i] += 1
                                    if bad_code_list[i] > 3:
                                        print('reinitializing codebook', i, 'usage', round(torch.mean((diff_idx == i)*1.).item(),3))
                                        with torch.no_grad():
                                            # model_mlp.codebook[j] = model_mlp.codebook[i].clone() + torch.randn(n_latent, requires_grad=True).to(device) * 0.0001
                                            model_mlp.codebook[i] = diff[i] # reinit to an actual processed embedding (pre quantization)
                                        bad_code_list[i] = 0


                            # elif epoch < n_epochs*2/3: # don't do this late in training
                            #     reinit_count = 0
                            #     for i in range(codebook_size):
                            #         # print('codebook', i, 'usage', round(torch.mean((diff_idx == i)*1.).item(),3))
                            #         # if a code is 0, reinitialize it
                            #         if torch.mean((diff_idx == i)*1.).item() < 0.0001 or torch.mean((diff_idx == i)*1.).item() > 0.5:
                            #             if i not in bad_code_list:
                            #                 bad_code_list[i] = 0
                            #             bad_code_list[i] += 1
                            #             if bad_code_list[i] > 3:
                            #                 # print('reinitializing codebook', i, 'bad count', bad_code_list[i])
                            #                 # reinitialize codebook
                            #                 reinit_count += 1
                            #                 with torch.no_grad():
                            #                     model_mlp.codebook[i] = torch.randn(n_latent, requires_grad=True)
                            #                     # print('reinitializing codebook', i)
                            #                 bad_code_list[i] = 0
                            #         else:
                            #             bad_code_list[i] = 0
                            #     print('reinit_count', reinit_count)


                        # save codebook usage as plot
                        fig, axs = plt.subplots(1, 1, figsize=(20,5))
                        axs.bar(range(codebook_size), [torch.mean((diff_idx == i)*1.).item() for i in range(codebook_size)])
                        axs.set_title('codebook usage')
                        axs.set_xlabel('codebook index')
                        axs.set_ylabel('usage')
                        fig.savefig(save_path_viz + 'codebook_usage_codes'+str(codebook_size)+'_intensity'+str(intensity)+'_actionprop'+str(action_prop)+'_k_data_aug'+str(k_data_aug)+'.png')
                        plt.close(fig)

                x_curr_test, x_next_test, a_curr_test, a_allowed_test = dataset.__getitem__(0, frame_i=2)
                reconstruct(x_curr_test.clone().to(device), x_next_test.clone().to(device), a_curr_test.clone().to(device), save_name='post_recon_frame_1.png')
                x_curr_test, x_next_test, a_curr_test, a_allowed_test = dataset.__getitem__(1, frame_i=5)
                reconstruct(x_curr_test.clone().to(device), x_next_test.clone().to(device), a_curr_test.clone().to(device), save_name='post_recon_frame_2.png')
                x_curr_test, x_next_test, a_curr_test, a_allowed_test = dataset.__getitem__(2, frame_i=1)
                reconstruct(x_curr_test.clone().to(device), x_next_test.clone().to(device), a_curr_test.clone().to(device), save_name='post_recon_frame_3.png')

                # save losses
                print(loss_all)
                np.save(save_path_viz + 'save_losses_codes'+str(codebook_size)+'_intensity'+str(intensity)+'_actionprop'+str(action_prop)+'_k_data_aug'+str(k_data_aug)+'.npy', np.array(loss_all))

